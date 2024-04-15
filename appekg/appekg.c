/**
* \file appekg.c
* \brief AppEKG Implementation
* 
* This current implementation supports two modes: storing data into 
* CSV files or using LDMS Streams to collect data.
*
* Environment variables -- see appekg.h header comment
*  
* Threading: basic idea: hash thread ID into a small range, use this
* as index into array of thread heartbeat data. If hash collides, ignore
* the 2nd+ threads and don't collect data on them.
**/

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

#define EKG_EXTERN
#include "appekg.h"

#ifdef INCLUDE_LDMS_STREAMS
#include "ldms/ldms_appstreams.h"
#endif

#define MAX_CSV_STRLEN 1024
#define MAX_JSON_STRLEN 2048
#define FILENAME_PREFIX "appekg"

//--------------------------------------------------------------------
// macro for getting thread ID; currently just uses pthread_self(),
// but this may not work on all platforms; e.g., some OpenMP
// implementations may not use pthreads under the hood. We eventually
// need to support compile options here.
#define THREAD_ID (((unsigned int)pthread_self()) % EKG_MAX_THREADS)
#define THREAD_ID_FULL ((unsigned int)pthread_self())
//#define THREAD_ID (omp_get_thread_num()%EKG_MAX_THREADS)
//#define THREAD_ID_FULL (omp_get_thread_num())
//--------------------------------------------------------------------

static enum {
    APPEKG_STAT_OK = 0,
    APPEKG_STAT_ERROR,
    APPEKG_STAT_DISABLED
} appekgStatus = APPEKG_STAT_DISABLED; /** status of APPEKG **/

// Output modes: Default CSV should always be compiled in as an
// available option; others should be ifdef'd to be selected for
// compilation; still need to create an output selection mode (and
// may need to move this into appekg.h for macro availability)
static enum {
    DO_CSV_FILES = 0,
#ifdef INCLUDE_LDMS_STREAMS
    DO_LDMS_STREAMS,
#endif
    DO_NO_OUTPUT
} ekgOutputMode = DO_CSV_FILES;

// Metadata generic metric structure; this is left over from
// some code that was integrated into LDMS, not sure if we should
// abandon it or not. We could certainly simplify it quite a bit
// by not supporting all the data types we aren't using.
#define MAX_BASE_METRICS 10
static struct Metric {
    char* name;
    enum {
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT,
        DOUBLE,
        STRING
    } type;
    union {
        uint8_t u8val;
        uint16_t u16val;
        uint32_t u32val;
        uint64_t u64val;
        int8_t i8val;
        int16_t i16val;
        int32_t i32val;
        int64_t i64val;
        float fpval;
        double dpval;
        //char    strval[64];  // leave out until we need it, too much space
    } v;
} baseMetrics[MAX_BASE_METRICS],
      threadMetrics[EKG_MAX_THREADS][EKG_MAX_HEARTBEATS * 2 + 2];

static int baseCount = 0;   /* number of base metrics */
static int metricCount = 0; /* number of metrics */

/* begin time for hbeats, per thread and per heartbeat, in microseconds */
static unsigned long beginHBTime[EKG_MAX_THREADS][EKG_MAX_HEARTBEATS] = {0};
static int numHeartbeats = 0; /* number of hbeats */
static char** hbNames;        // heartbeat names

// thread id for the appekg sampling thread
static pthread_t tid = 0;

static unsigned long samplingInterval = 1; /* in seconds */
static int doSampling = 0;
static FILE* csvFH = 0;
static void* performSampling(void* arg);
static int allowStderr = 0;
static pthread_mutex_t hblock;
static struct timespec programStartTime;
static unsigned int applicationID, jobID;
static char jsonFilename[1024];

/**
 * \brief Initialize AppEKG
 *
 * All AppEKG initialization: data structures, job/run data, output files,
 * sampling thread startup, etc.
 *
 * \param appekgEnable enable data sampling 0-disable 1-enable.
 * \param appid is a unique identifying number for this application
 * \param jobid is the job id # (0 means look at PBS_JOBID env var)
 * \param rank is the MPI (or other) process rank designation
 * \param silent is nonzero if want no stderr
 * \return 0 on success. -1 is returned on failure.
 */
int ekgInitialize(unsigned int pNumHeartbeats, float pSamplingInterval,
                  unsigned int appid, unsigned int jobid, unsigned int rank,
                  unsigned int silent)
{
    int i;
    char* p;
    if (appekgStatus == APPEKG_STAT_OK)
        return 0; // other thread already did the init?
    pthread_mutex_lock(&hblock);
    if (appekgStatus == APPEKG_STAT_OK) {
        pthread_mutex_unlock(&hblock);
        return 0; // other thread already did the init?
    }
    allowStderr = !silent;
    applicationID = appid;

    clock_gettime(CLOCK_REALTIME, &programStartTime);

    /* get number of hbeats from environment variable */
    // JEC: I disabled this, is just a built-in init param; this
    // should not be changeable in the environment
    //p = getenv("NO_OF_HBEATS");
    //numberOfHB = atoi(p);
    numHeartbeats = pNumHeartbeats;
    // set up heartbeat name array (index 0 is not used)
    hbNames = (char**)calloc(sizeof(char*), numHeartbeats + 1);

    /* check for sampling interval from environment variable */
    samplingInterval = pSamplingInterval;
    p = getenv("APPEKG_SAMPLING_INTERVAL");
    if (p)
        samplingInterval = strtoul(p, 0, 10);

    // Per thread heartbeat data; these are accessible in the
    // application instrumentation macros, and so their names
    // must be namespaced and not conflict with application names
    _ekgHBEndFlag = (unsigned int*)calloc(sizeof(unsigned int),
                                          EKG_MAX_HEARTBEATS * EKG_MAX_THREADS);
    _ekgHBCount = (unsigned int*)calloc(sizeof(unsigned int),
                                        EKG_MAX_HEARTBEATS * EKG_MAX_THREADS);
    _ekgActualThreadID =
          (unsigned int*)calloc(sizeof(unsigned int), EKG_MAX_THREADS);
    _ekgThreadId = pthread_self;

    // Set up job id if needed: JEC - I changed this to override the
    // initialization argument rather than defer to it.
    int jid = 0;
    char* js = getenv("PBS_JOBID");
    if (js)
        jid = (int)strtol(js, 0, 10);
    else {
        js = getenv("SLURM_JOB_ID");
        if (js)
            jid = (int)strtol(js, 0, 10);
    }
    if (jid > 0)
        jobid = jid;
    jobID = jobid;

    // set up component id from hostname, if hostname has a number in it
    int cid = 0;
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        char* idp = strpbrk(hostname, "0123456789");
        if (idp)
            cid = (int)strtol(idp, 0, 10);
    }

    // initialize base metrics
    baseMetrics[baseCount].name = "timemsec";
    baseMetrics[baseCount].type = UINT64;
    baseMetrics[baseCount].type = UINT64;
    baseMetrics[baseCount++].v.i64val = 0;
    baseMetrics[baseCount].name = "component";
    baseMetrics[baseCount].type = UINT32;
    baseMetrics[baseCount++].v.i32val = cid;
    baseMetrics[baseCount].name = "appid";
    baseMetrics[baseCount].type = UINT32;
    baseMetrics[baseCount++].v.i32val = appid;
    baseMetrics[baseCount].name = "jobid";
    baseMetrics[baseCount].type = UINT32;
    baseMetrics[baseCount++].v.i32val = jobid;
    baseMetrics[baseCount].name = "rank";
    baseMetrics[baseCount].type = UINT32;
    baseMetrics[baseCount++].v.i32val = rank;
    baseMetrics[baseCount].name = "pid";
    baseMetrics[baseCount].type = UINT32;
    baseMetrics[baseCount++].v.i32val = getpid();

    // initialize thread metrics
    metricCount = 0;
    threadMetrics[0][metricCount].name = strdup("threadID");
    threadMetrics[0][metricCount].v.u64val = 0;
    threadMetrics[0][metricCount++].type = UINT64;
    // declare metrics for hbeat count and time
    for (i = 1; i <= numHeartbeats; i++) {
        char buffer[48];
        snprintf(buffer, sizeof(buffer) - 2, "hbcount%d", i);
        // initialize for thread index 0, we'll use others later?
        // when writing CSV header, we only need one copy, so only
        // put names in thread index 0
        threadMetrics[0][metricCount].name = strdup(buffer);
        threadMetrics[0][metricCount].v.u64val = 0;
        threadMetrics[0][metricCount++].type = UINT64;
        snprintf(buffer, sizeof(buffer) - 2, "hbduration%d", i);
        threadMetrics[0][metricCount].name = strdup(buffer);
        threadMetrics[0][metricCount].v.dpval = 0.0;
        threadMetrics[0][metricCount++].type = DOUBLE;
    }
    appekgStatus = APPEKG_STAT_OK;
    // start up sampling thread
    doSampling = 1;
    pthread_mutex_unlock(&hblock);
    pthread_create(&tid, 0, performSampling, 0);
    return 0;
}

// forward declaration for compiler
static void finalizeHeartbeatData(void* v);

/**
* \brief Finalize AppEKG
* \return 0 on success. negative value is returned on failure.
**/
void ekgFinalize(void)
{
    /* if never initialized, don't clean up */
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    pthread_mutex_lock(&hblock);
    if (appekgStatus != APPEKG_STAT_OK) {
        pthread_mutex_unlock(&hblock);
        return;
    }
    doSampling = 0;
    pthread_mutex_unlock(&hblock);
    finalizeHeartbeatData((void*)0);
    appekgStatus = APPEKG_STAT_DISABLED;
    if (_ekgHBEndFlag) {
        free(_ekgHBEndFlag);
        _ekgHBEndFlag = 0;
    }
    if (_ekgHBCount) {
        free(_ekgHBCount);
        _ekgHBCount = 0;
    }
    return;
}

/**
* \brief Stop AppEKG data collection temporarily (TODO)
**/
void ekgDisable(void)
{
    // TODO
}

/**
* \brief Start or restart AppEKG data collection (TODO)
**/
void ekgEnable(void)
{
    // TODO
}

/**
* \brief Begin a heartbeat
*
* Creates a timestamp to begin this heartbeat; although it does
* not do any locking, it should be thread safe because each thread
* has its own data, and thread hash collisions are handled before
* setting data.
*
* \param id is the heartbeat ID
*/
void ekgBeginHeartbeat(unsigned int hbId)
{
    if (hbId <= 0 || hbId > numHeartbeats)
        return;
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    unsigned int thId = THREAD_ID;
    unsigned int realId = THREAD_ID_FULL;
    if (_ekgActualThreadID[thId] == 0)
        _ekgActualThreadID[thId] = realId;
    else if (_ekgActualThreadID[thId] != realId)
        return; // collision and we didn't win, so leave
    struct timespec start;
    if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
        if (allowStderr)
            perror("AppEKG: clock gettime error");
        return;
    }
    // subtract off program start time seconds (skip nanosec)
    start.tv_sec -= programStartTime.tv_sec;
    // changed to be microseconds; nanoseconds might overflow an int
    // TODO: still think about this, maybe microseconds will overflow, too
    beginHBTime[thId][hbId] =
          ((1000000 * start.tv_sec) + (start.tv_nsec / 1000));
    return;
}

/**
* \brief End a heartbeat
* 
* This function timestamps the end of the current heartbeat, 
* and then updates the calculation of the count and durations
* of the occurrences of this heartbeat.
* TODO: make this simpler!
*
* \param id is the heartbeat ID
**/
void ekgEndHeartbeat(unsigned int hbId)
{
    double duration, avg;
    if (hbId <= 0 || hbId > numHeartbeats)
        return;
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    struct timespec endTime;
    unsigned int thId = THREAD_ID;
    if (_ekgActualThreadID[thId] != THREAD_ID_FULL)
        return;
    if (clock_gettime(CLOCK_REALTIME, &endTime) == -1) {
        if (allowStderr)
            perror("AppEKG: clock gettime error");
        return;
    }
    // subtract off program start time seconds (skip nanosec)
    endTime.tv_sec -= programStartTime.tv_sec;
    // convert secs to microsecond, see comment in beginHB
    unsigned long endHBTime =
          ((1000000 * endTime.tv_sec) + (endTime.tv_nsec / 1000));
    // calculate duration
    duration = (endHBTime - beginHBTime[thId][hbId]);
    // lock count and duration update from other threads
    pthread_mutex_lock(&hblock);
    hbId = (hbId - 1) * 2 + 1; // map to thread data array
    if (threadMetrics[thId][hbId].v.u64val > 0) {
        // too much arithmetic for high-intensity heartbeats, this
        // could be simplified if we just kept the duration sum
        avg = ((threadMetrics[thId][hbId + 1].v.dpval *
                      (threadMetrics[thId][hbId].v.u64val) +
                duration) /
               (threadMetrics[thId][hbId].v.u64val + 1));
        threadMetrics[thId][hbId + 1].v.dpval = avg;
        threadMetrics[thId][hbId].v.u64val++;
    } else {
        threadMetrics[thId][hbId + 1].v.dpval = duration;
        threadMetrics[thId][hbId].v.u64val++;
    }
    pthread_mutex_unlock(&hblock);
    return;
}

/**
* \brief Provide a name for a heartbeat
*
* Give AppEKG a name for a heartbeat; AppEKG will store its own copy.
*
* \param id is the heartbeat ID
* \param name is the string name
* \return 0 on success, -1 on failure
**/
int ekgNameHeartbeat(unsigned int id, char* name)
{
    if (id < 1 || id > numHeartbeats)
        return -1;
    if (hbNames[id] != 0)
        free(hbNames[id]);
    hbNames[id] = strdup(name);
    return 0;
}

/**
* \brief Find ID of heartbeat from name
*
* Find ID of the given named heartbeat
*
* \param name is the string name
* \return ID value on success, 0 on failure
**/
unsigned int ekgIdOfHeartbeat(char* name)
{
    int i;
    // valid IDs start at 1, index 0 is not used
    for (i = 1; i <= numHeartbeats; i++) {
        if (hbNames[i] == 0)
            continue;
        if (!strcmp(hbNames[i], name))
            return i;
    }
    return 0;
}

/**
* \brief Find name of heartbeat from ID
*
* Ask for stored name of heartbeat; returns NULL if none, ptr to 
* new copy of string if found.
*
* \param id is the heartbeat ID
* \return new allocated copy of name on success, 0 on failure
**/
char* ekgNameOfHeartbeat(unsigned int id)
{
    if (id < 1 || id > numHeartbeats || hbNames[id] == 0)
        return 0;
    return strdup(hbNames[id]);
}

#ifdef INCLUDE_LDMS_STREAMS
/**
* \brief Output hearbeat datapoint LDMS stream (broken!)
*
* This needs completely redone for threaded data. Currently not used.
**/
static void outputLDMSStreamsData()
{
    char str[MAX_JSON_STRLEN]; // DANGEROUS! Use snprintf!
    int j = 0;
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    j += sprintf(&str[j], "%s", "{");
    for (i = 0; i < metricCount; i++) {
        if (i > 0)
            j += sprintf(&str[j], ", ");
        switch (metrics[i].type) {
        case UINT8:
            j += sprintf(&str[j], " \"%s\":%u, ", metrics[i].name,
                         metrics[i].v.u8val);
            break;
        case UINT16:
            j += sprintf(&str[j], "\"%s\":%u", metrics[i].name,
                         metrics[i].v.u16val);
            break;
        case UINT32:
            j += sprintf(&str[j], "\"%s\":%u", metrics[i].name,
                         metrics[i].v.u32val);
            break;
        case UINT64:
            j += sprintf(&str[j], "\"%s\":%lu", metrics[i].name,
                         metrics[i].v.u64val);
            break;
        case INT8:
            j += sprintf(&str[j], "\"%s\":%d", metrics[i].name,
                         metrics[i].v.i8val);
            break;
        case INT16:
            j += sprintf(&str[j], "\"%s\":%d", metrics[i].name,
                         metrics[i].v.i16val);
            break;
        case INT32:
            j += sprintf(&str[j], "\"%s\":%d", metrics[i].name,
                         metrics[i].v.i32val);
            break;
        case INT64:
            j += sprintf(&str[j], "\"%s\":%ld", metrics[i].name,
                         metrics[i].v.i64val);
            break;
        case FLOAT:
            j += sprintf(&str[j], "\"%s\":%g", metrics[i].name,
                         metrics[i].v.fpval);
            break;
        case DOUBLE:
            j += sprintf(&str[j], "\"%s\":%g", metrics[i].name,
                         metrics[i].v.dpval);
            break;
        case STRING:
            j += sprintf(&str[j], "\"%s\":\"%s\"", metrics[i].name,
                         metrics[i].v.strval);
            break;
        default:
            j += sprintf(&str[j], "\"%s\":\"%s\"", metrics[i].name, "0");
            break;
        }
    }
    j += sprintf(&str[j], "%s", "}");
    ldms_appinst_publish(str);
}
#endif // INCLUDE_LDMS_STREAMS

/**
* \brief Write Metadata out in JSON format
*
* TODO: add more fields, possibly number of active threads, sampling interval,
*       more? Get some MPI env variables, number of processes, etc.
* page with environment variables: https://hpcc.umd.edu/hpcc/help/slurmenv.html
* number of nodes: SLURM_JOB_NUM_NODES, PBS_NUM_NODES
* number of procs: SLURM_NTASKS, PBS_NP
* end time: ?? need another function to add to JSON data
**/
static char* metadataToJSON()
{
    static char tdatastr[2048]; // static to enable return; not thread safe
    int i, tstrlen = 0;
    tstrlen +=
          snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1, "{\n");
    // i starts at 1 to skip timestamp, but include component, appid, jobid,
    // rank, and pid
    for (i = 1; i < baseCount; i++) {
        if (i > 1)
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, ",\n");
        switch (baseMetrics[i].type) {
        case UINT8:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%u",
                                baseMetrics[i].name, baseMetrics[i].v.u8val);
            break;
        case UINT16:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%u",
                                baseMetrics[i].name, baseMetrics[i].v.u16val);
            break;
        case UINT32:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%u",
                                baseMetrics[i].name, baseMetrics[i].v.u32val);
            break;
        case UINT64:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%lu",
                                baseMetrics[i].name, baseMetrics[i].v.u64val);
            break;
        case INT8:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%d",
                                baseMetrics[i].name, baseMetrics[i].v.i8val);
            break;
        case INT16:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%d",
                                baseMetrics[i].name, baseMetrics[i].v.i16val);
            break;
        case INT32:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%d",
                                baseMetrics[i].name, baseMetrics[i].v.i32val);
            break;
        case INT64:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%ld",
                                baseMetrics[i].name, baseMetrics[i].v.i64val);
            break;
        case FLOAT:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%g",
                                baseMetrics[i].name, baseMetrics[i].v.fpval);
            break;
        case DOUBLE:
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, "\"%s\":%g",
                                baseMetrics[i].name, baseMetrics[i].v.dpval);
            break;
        //case STRING: fprintf(csvFH,"%s",baseMetrics[i].v.strval); break;
        default:
            tstrlen +=
                  snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                           "\"%s\":\"unknown\"", baseMetrics[i].name);
            break;
        }
    }
    // Output some extra information
    char hostname[48];
    struct utsname unameInfo;
    tstrlen +=
          snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                   ",\n\"starttime\":%u", (unsigned)programStartTime.tv_sec);
    gethostname(hostname, sizeof(hostname) - 1);
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"hostname\":\"%s\"", hostname);
    uname(&unameInfo);
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"osname\":\"%s\"", unameInfo.sysname);
    // nodename is same as hostname (but hostname may be more reliable)
    //tstrlen += snprintf(tdatastr+tstrlen, sizeof(tdatastr)-tstrlen-1,
    //                    ",\n\"nodename\":\"%s\"", unameInfo.nodename);
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"osrelease\":\"%s\"", unameInfo.release);
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"osversion\":\"%s\"", unameInfo.version);
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"architecture\":\"%s\"", unameInfo.machine);
    // try to get number of nodes from job scheduler
    char* ns = getenv("PBS_NUM_NODES");
    int numnodes = 0;
    if (ns)
        numnodes = (int)strtol(ns, 0, 10);
    else {
        ns = getenv("SLURM_JOB_NUM_NODES");
        if (ns)
            numnodes = (int)strtol(ns, 0, 10);
    }
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"numnodes\":%d", numnodes);
    // try to get number of processes (ranks) from job scheduler
    char* ps = getenv("PBS_NP");
    int numprocs = 0;
    if (ps)
        numprocs = (int)strtol(ps, 0, 10);
    else {
        ps = getenv("SLURM_NTASKS");
        if (ps)
            numprocs = (int)strtol(ps, 0, 10);
    }
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"numprocs\":%d", numprocs);
    // heartbeat names are put in an array
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        ",\n\"hbnames\":{");
    for (i = 1; i <= numHeartbeats; i++) {
        if (hbNames[i] != 0) {
            if (i > 1)
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, ",");
            tstrlen +=
                  snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                           "\"%d\":\"%s\"", i, hbNames[i]);
        }
    }
    tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                        "}\n}\n");
    return tdatastr;
}

/**
* \brief Output headers of CSV data files
*
* TODO: this also creates the metadata file (at the end), and so
* this routine probably should be renamed, and/or refactored
**/
static void writeCSVHeaders()
{
    int i;
    char* s;
    char fullFilename[1024], pathFormat[800];
    FILE* mf;
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    // build output data path
    strcpy(fullFilename, ".");
    s = getenv("APPEKG_OUTPUT_PATH");
    if (s) {
        strncpy(pathFormat, s, sizeof(pathFormat));
        pathFormat[sizeof(pathFormat) - 1] = '\0';
        // convert any %s formats into ss, for security
        s = pathFormat;
        while ((s = strstr(s, "%s")) != NULL) {
            *(s + 1) = 'd';
            s++;
        }
        // count # of %s in format; use it if two or less
        s = pathFormat;
        i = 0;
        while ((s = index(s, '%')) != NULL) {
            i++;
            s++;
        }
        if (i <= 2)
            snprintf(fullFilename, sizeof(fullFilename), pathFormat,
                     applicationID, jobID);
        else
            strcpy(fullFilename, pathFormat);
    }
    // copy path prefix back to pathFormat to use twice
    strcpy(pathFormat, fullFilename);
    // try mkdir just in case; this could create the "jobID" directory
    // if it is last and the format specified it
    mkdir(pathFormat, 0770);
    // name each rank's data file with PID
    sprintf(fullFilename, "%s/%s-%d.csv", pathFormat, FILENAME_PREFIX,
            getpid());
    csvFH = fopen(fullFilename, "w");
    if (!csvFH) {
        // try to put data files in current directory
        strcpy(pathFormat, ".");
        sprintf(fullFilename, "%s/%s-%d.csv", pathFormat, FILENAME_PREFIX,
                getpid());
        csvFH = fopen(fullFilename, "w");
    }
    if (!csvFH) {
        if (allowStderr)
            fprintf(stderr, "AppEKG: cannot open data file...disabling");
        appekgStatus = APPEKG_STAT_DISABLED;
        return;
    }
    // write out column headers (CHANGED: only output first one (time)
    // when we finalize this, we can remove the loop
    for (i = 0; i < baseCount; i++) {
        if (i > 0)
            break; // stop after time (was fprintf(csvFH,",");)
        fprintf(csvFH, "%s", baseMetrics[i].name);
    }
    // heartbeats have two metrics each, count and duration, and start at 1
    for (i = 0; i < metricCount; i++) {
        fprintf(csvFH, ",");
        // decision: don't print HB custom names in CSV header
        // - can remove this code when this is a final decision
        //if (i>0 && hbNames[(i-1)/2+1] != 0)
        //   fprintf(csvFH,"%s-%s",threadMetrics[0][i].name,hbNames[(i-1)/2+1]);
        //else
        fprintf(csvFH, "%s", threadMetrics[0][i].name);
    }
    fprintf(csvFH, "\n");
    sprintf(jsonFilename, "%s/%s-%d.json", pathFormat, FILENAME_PREFIX,
            getpid());
    mf = fopen(jsonFilename, "w");
    fputs(metadataToJSON(), mf);
    fclose(mf);
}

/**
* \brief Output heartbeat datapoint to CSV file
*
* This routine collects all heartbeat data per thread into a
* single string of CSV data, then chooses to output it or not,
* depending on whether all threads had 0 data or not. DONE: this
* needs some thinking, we might rather identify the "active" threads
* first, then always output data lines for the active threads, whether
* any heartbeats were non-zero or not.
**/
static void outputCSVData()
{
    int i, tstrlen = 0; //, allzeros;
    unsigned int tid;
    char tdatastr[MAX_CSV_STRLEN];
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    //fprintf(csvFH,"CSV Output\n");
    pthread_mutex_lock(&hblock);
    // if output hasn't started yet, write header data
    if (csvFH == 0)
        writeCSVHeaders();
    // write headers opens the files; if unsuccessful, exit
    if (appekgStatus != APPEKG_STAT_OK) {
        pthread_mutex_unlock(&hblock);
        return;
    }
    for (tid = 0; tid < EKG_MAX_THREADS; tid++) {
        //fprintf(csvFH,"Thread %d\n",tid);
        // new: if no thread is registered in this slot, continue,
        // else create and print a data record regardless of whether
        // it is all zeroes or not
        if (_ekgActualThreadID[tid] == 0)
            continue;
        tstrlen = 0;
        // allzeros = 1;
        // CHANGED: stop after time (first base metric)
        // when we finalize this, we can remove the loop
        for (i = 0; i < baseCount; i++) {
            if (i > 0)
                break; // Stop after first metric
            //tstrlen += snprintf(tdatastr+tstrlen, sizeof(tdatastr)-tstrlen-1,
            //                    ",");
            switch (baseMetrics[i].type) {
            case UINT8:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%u",
                                    baseMetrics[i].v.u8val);
                break;
            case UINT16:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%u",
                                    baseMetrics[i].v.u16val);
                break;
            case UINT32:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%u",
                                    baseMetrics[i].v.u32val);
                break;
            case UINT64:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%lu",
                                    baseMetrics[i].v.u64val);
                break;
            case INT8:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%d",
                                    baseMetrics[i].v.i8val);
                break;
            case INT16:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%d",
                                    baseMetrics[i].v.i16val);
                break;
            case INT32:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%d",
                                    baseMetrics[i].v.i32val);
                break;
            case INT64:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%ld",
                                    baseMetrics[i].v.i64val);
                break;
            case FLOAT:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%g",
                                    baseMetrics[i].v.fpval);
                break;
            case DOUBLE:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%g",
                                    baseMetrics[i].v.dpval);
                break;
            //case STRING: fprintf(csvFH,"%s",baseMetrics[i].v.strval); break;
            default:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "unknown");
                break;
            }
        }
        //allzeros = 1;
        tstrlen += snprintf(tdatastr + tstrlen, sizeof(tdatastr) - tstrlen - 1,
                            ",%d", tid);
        // Currently, heartbeat counts and time start at index 1, not 0,
        // because it makes for easier (faster) math in begin/end heartbeat
        for (i = 1; i < metricCount; i++) {
            tstrlen += snprintf(tdatastr + tstrlen,
                                sizeof(tdatastr) - tstrlen - 1, ",");
            //if (threadMetrics[tid][i].v.dpval != 0.0)
            //   allzeros = 0;
            switch (threadMetrics[0][i].type) {
            case UINT8:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%u",
                                    threadMetrics[tid][i].v.u8val);
                break;
            case UINT16:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%u",
                                    threadMetrics[tid][i].v.u16val);
                break;
            case UINT32:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%u",
                                    threadMetrics[tid][i].v.u32val);
                break;
            case UINT64:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%lu",
                                    threadMetrics[tid][i].v.u64val);
                break;
            case INT8:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%d",
                                    threadMetrics[tid][i].v.i8val);
                break;
            case INT16:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%d",
                                    threadMetrics[tid][i].v.i16val);
                break;
            case INT32:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%d",
                                    threadMetrics[tid][i].v.i32val);
                break;
            case INT64:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%ld",
                                    threadMetrics[tid][i].v.i64val);
                break;
            case FLOAT:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%.3f",
                                    (double)threadMetrics[tid][i].v.fpval);
                break;
            case DOUBLE:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "%.3f",
                                    threadMetrics[tid][i].v.dpval);
                break;
            //case STRING: fprintf(csvFH,"%s",threadMetrics[tid][i].v.strval);
            //  break;
            default:
                tstrlen += snprintf(tdatastr + tstrlen,
                                    sizeof(tdatastr) - tstrlen - 1, "unknown");
                break;
            }
            // Zero out all heartbeat data (but not base metrics)
            threadMetrics[tid][i].v.dpval = 0.0;
        }
        //if (!allzeros)
        fprintf(csvFH, "%s\n", tdatastr);
    }                              // end outer for
    pthread_mutex_unlock(&hblock); // unlock after all HB data
    fflush(csvFH);
}

/**
* \brief Output hearbeat data to CSV file or LDMS stream
*
* This routine writes out a sampled datapoint; it was recently
* refactored to not directly include code for each output data
* format, and needs more work to re-include LDMS streams, and
* other data sinks. TODO: Should allow both (and more) to be 
* compiled in, runtime selected
**/
static void outputHeartbeatData()
{
    struct timeval curtime;
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    // set current sample's timestamp
    gettimeofday(&curtime, 0);
    curtime.tv_sec -= programStartTime.tv_sec;
    baseMetrics[0].v.i64val =
          (curtime.tv_sec * 1000) + (curtime.tv_usec / 1000);
    //baseMetrics[0].v.i64val = curtime.tv_sec;
    //baseMetrics[1].v.i64val = curtime.tv_usec;
    if (ekgOutputMode == DO_CSV_FILES)
        outputCSVData();
    // TODO Other choices here
}

/**
* \brief Finish AppEKG data collection
*
* This writes one last datapoint out (perhaps it should not?)
* and closes the output mechanism.
*
* \param arg is not used (but required by pthreads)
**/
static void finalizeHeartbeatData(void* arg)
{
    if (appekgStatus != APPEKG_STAT_OK)
        return;
    outputHeartbeatData();
    if (ekgOutputMode == DO_CSV_FILES) {
        fclose(csvFH);
        csvFH = 0;
        FILE* jfh = fopen(jsonFilename, "r+");
        if (jfh) {
            fseek(jfh, -3, SEEK_END);
            struct timespec curtime;
            clock_gettime(CLOCK_REALTIME, &curtime);
            fprintf(jfh, ",\n\"endtime\":%u,\n\"duration\":%u\n}\n",
                    (unsigned)curtime.tv_sec,
                    (unsigned)(curtime.tv_sec - programStartTime.tv_sec));
            fclose(jfh);
        }
    }
}

/**
* \brief Main routine for data-collecting thread
*
* This routine is invoked as the beginning of a pthread-created
* thread, and runs for the duration of the application. It samples
* the collected metric data at the desired sampling rate, and invokes
* the output function.
*
* \param arg is not used
* \return always returns 0
**/
static void* performSampling(void* arg)
{
    pthread_detach(pthread_self());
    pthread_cleanup_push((&finalizeHeartbeatData), ((void*)0));
    while (doSampling) {
        sleep(samplingInterval);
        outputHeartbeatData();
    }
    pthread_cleanup_pop(1);
    return 0;
}
