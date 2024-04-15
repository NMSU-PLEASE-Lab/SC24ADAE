/**
* \mainpage  AppEKG
*
* AppEKG is a heartbeat instrumentation library and analysis framework.
*
* Multiple heartbeats are distinguished by their different IDs, which
* are integer identifiers starting at 1. Heartbeats can be named through
* the API. For instrumentation sites with high activity, an integer rate
* factor can be applied; e.g., a rate factor of 10 will cause a heartbeat
* once every 10 executions of the instrumentation site.
*
* The defined macros are the preferred way of creating AppEKG
* instrumentation in an application, but the underlying functions
* are also available in the API; the functions for begin/end
* heatbeats do not inherently support a rate factor, however. 
*
* The macro interface is:
* \li EKG_BEGIN_HEARTBEAT(id, rateFactor) 
* \li EKG_END_HEARTBEAT(id) 
* \li EKG_PULSE_HEARTBEAT(id, rateFactor) 
* \li EKG_INITIALIZE(numHeartbeats, samplingInterval, appid, jobid, rank, silent) 
* \li EKG_FINALIZE() 
* \li EKG_DISABLE() 
* \li EKG_ENABLE() 
* \li EKG_NAME_HEARTBEAT(id, name) 
* \li EKG_IDOF_HEARTBEAT(name) 
* \li EKG_NAMEOF_HEARTBEAT(id) 
*
* Heartbeat IDs are small integers starting at 1, and should be sequential.
* A unique heartbeat ID is meant to represent a particular phase or kernel 
* of the application, and generally each instrumentation site has a unique
* heartbeat ID.
*
* rateFactor controls the speed of heartbeat production if an instrumentation
* site is invoked too frequently. A rateFactor of 100, for example, would 
* produce a heartbeat once every 100 executions of the instrumentation site.
*
* AppEKG initialization accepts as parameters the number of unique heartbeats
* (maximum heartbeat ID), the number of seconds between data samples, a
* unique application ID, job ID, MPI rank, and a silent flag for turning
* off (unlikely but possible) stderr messages.
*
* Heartbeats can be given a name using the API; names should generally
* refer to the conceptual meaning of the application phase or kernel it
* is capturing.
*
* Environment Variables:
* APPEKG_SAMPLING_INTERVAL : integer, number of seconds between samples; will
*                            override AppEKG initialization parameter
* APPEKG_OUTPUT_PATH : string to prepend to output file names; '/' is added 
*                      at end and does not need to be included. If string
*                      contains one %d, the application ID is inserted at
*                      that spot; if it contains two %d's, the job ID is
*                      inserted for the second one. Other % options will 
*                      cause the string to be used as is, without 
*                      substitutions.
* PBS_JOBID : if found, used for the 'jobid' data field, if param jobid=0
* SLURM_JOB_ID : if found, used for the 'jobid' data field, if param jobid=0
**/

// TODO: Should PBS/SLURM ids override initialization parameter?

/**
* \file appekg.h
*
* \brief External API Definition
**/
#ifndef __APPEKG_H__
#define __APPEKG_H__

//#include <inttypes.h>
//#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef APPEKG_DISABLE

//--------------------------------------------------------------------
// Completely empy AppEKG macro definitions, for compiling AppEKG out
//--------------------------------------------------------------------
#define EKG_BEGIN_HEARTBEAT(id, rateFactor)
#define EKG_END_HEARTBEAT(id)
#define EKG_PULSE_HEARTBEAT(id, rateFactor)
#define EKG_INITIALIZE(numHeartbeats, samplingInterval, appid, jobid, rank,    \
                       silent)
#define EKG_FINALIZE()
#define EKG_DISABLE()
#define EKG_ENABLE()
#define EKG_NAME_HEARTBEAT(id, name)
#define EKG_IDOF_HEARTBEAT(name)
#define EKG_NAMEOF_HEARTBEAT(id)

#else

// Operating Parameters
#define EKG_MAX_THREADS 57
#define EKG_MAX_HEARTBEATS 20

//--------------------------------------------------------------------
// API Macros
// TODO: technically, the actual thread id check and set in the begin
// macro does have a race condition, but we really don't want the
// overhead of a lock; the race would be only over those threads that
// actually collided in the hashing, and it is only a few instructions
// long. The first heartbeat might get screwed up but after that, things
// should be ok since a thread id is recorded at that point.
//--------------------------------------------------------------------

/**
* \brief Begin a heartbeat (insert this at beginning site of instrumentation)
*
* @param id is the heartbeat id (1 and up)
* @param rateFactor limits HB production (100 == 1 HB every 100 invocations)
**/
#define EKG_BEGIN_HEARTBEAT(id, rateFactor)                                    \
    do {                                                                       \
        unsigned int rid = _ekgThreadId();                                     \
        unsigned int tid = rid % EKG_MAX_THREADS;                              \
        if (_ekgActualThreadID[tid] == 0)                                      \
            _ekgActualThreadID[tid] = rid;                                     \
        else if (_ekgActualThreadID[tid] != rid)                               \
            break;                                                             \
        tid = tid * EKG_MAX_HEARTBEATS + (id)-1;                               \
        if ((_ekgHBCount[tid]++) % (rateFactor) == 0) {                        \
            ekgBeginHeartbeat((id));                                           \
            _ekgHBEndFlag[tid] = 1;                                            \
        }                                                                      \
    } while (0)

/**
* \brief End a heartbeat (insert this at ending site of instrumentation)
*
* @param id is the heartbeat id (1 and up)
**/
#define EKG_END_HEARTBEAT(id)                                                  \
    do {                                                                       \
        unsigned int rid = _ekgThreadId();                                     \
        unsigned int tid = rid % EKG_MAX_THREADS;                              \
        if (_ekgActualThreadID[tid] != rid)                                    \
            break;                                                             \
        tid = tid * EKG_MAX_HEARTBEATS + (id)-1;                               \
        if (_ekgHBEndFlag[tid]) {                                              \
            ekgEndHeartbeat((id));                                             \
            _ekgHBEndFlag[tid] = 0;                                            \
        }                                                                      \
    } while (0)

/**
* \brief Create an impulse heartbeat (single site instrumentation)
*
* @param id is the heartbeat id (1 and up)
* @param rateFactor limits HB production (100 == 1 HB every 100 invocations)
**/
#define EKG_PULSE_HEARTBEAT(id, rateFactor)                                    \
    do {                                                                       \
        unsigned int rid = _ekgThreadId();                                     \
        unsigned int tid = rid % EKG_MAX_THREADS;                              \
        if (_ekgActualThreadID[tid] == 0)                                      \
            _ekgActualThreadID[tid] = rid;                                     \
        else if (_ekgActualThreadID[tid] != rid)                               \
            break;                                                             \
        tid = tid * EKG_MAX_HEARTBEATS + (id)-1;                               \
        if ((_ekgHBCount[tid]++) % (rateFactor) == 0) {                        \
            ekgBeginHeartbeat((id));                                           \
            ekgEndHeartbeat((id));                                             \
        }                                                                      \
    } while (0)

/**
* \brief Initialize AppEKG
*
* @param enable is 1 to enable, 0 to disable
* @param numHeartbeats is the number of heartbeats from this application
* @param appid is a unique application ID for this application
* @param jobid is the job ID for this execution (0 == look for PBS/SLURM
*        environment variables)
* @param rank is the MPI rank of this process (can be 0)
* @param silent is 1 to turn off any AppEKG stderr reporting
**/
#define EKG_INITIALIZE(numHeartbeats, samplingInterval, appid, jobid, rank,    \
                       silent)                                                 \
    do {                                                                       \
        ekgInitialize((numHeartbeats), (samplingInterval), (appid), (jobid),   \
                      (rank), (silent));                                       \
    } while (0)

/**
* \brief Shut down AppEKG and stop collecting heartbeat data
**/
#define EKG_FINALIZE()                                                         \
    do {                                                                       \
        ekgFinalize();                                                         \
    } while (0)

/**
* \brief Stop AppEKG from collecting heartbeat data, possibly temporarily
**/
#define EKG_DISABLE()                                                          \
    do {                                                                       \
        ekgDisable();                                                          \
    } while (0)

/**
* \brief Start (restart) AppEKG collecting heartbeat data
**/
#define EKG_ENABLE()                                                           \
    do {                                                                       \
        ekgEnable();                                                           \
    } while (0)

/**
* \brief Provide a name for a heartbeat
*
* @param id is the heartbeat id (1 and up)
* @param name is the string name of the heartbeat (will be copied)
**/
#define EKG_NAME_HEARTBEAT(id, name)                                           \
    do {                                                                       \
        ekgNameHeartbeat((id), (name));                                        \
    } while (0)

/**
* \brief Lookup heartbeat by name and return the ID
*
* @param name is the string name of the heartbeat
**/
#define EKG_IDOF_HEARTBEAT(name) ({ekgIdOfHeartbeat((name))})

/**
* \brief Lookup heartbeat by ID and return the name
*
* @param id is the heartbeat id (1 and up)
* @return a newly allocated string name of heartbeat, or NULL
**/
#define EKG_NAMEOF_HEARTBEAT(id) ({ekgNameOfHeartbeat((id))})

//--------------------------------------------------------------------
// API functions
//--------------------------------------------------------------------

/**
* \brief Begin a heartbeat
*
* \param id is the heartbeat ID
*/
void ekgBeginHeartbeat(unsigned int id);

/**
* \brief End a heartbeat
*
* \param id is the heartbeat ID
*/
void ekgEndHeartbeat(unsigned int id);

void ekgPulseHeartbeat(unsigned int id);

/**
* \brief Initialize AppEKG for heartbeat collection
*
* \param appekg_enable enable data sampling 0-disable 1-enable.
* \param appid is a unique identifying number for this application
* \param jobid is the job id # (0 means look at PBS_JOBID env var)
* \param rank is the MPI (or other) process rank designation
* \param silent is nonzero if want no stderr
* \return 0 on success, negative on failure
**/
int ekgInitialize(unsigned int numHeartbeats, float samplingInterval,
                  unsigned int appid, unsigned int jobid, unsigned int rank,
                  unsigned int silent);

/**
* \brief Finish AppEKG data collection, permanently
**/
void ekgFinalize(void);

/**
* \brief Stop AppEKG data collection temporarily
**/
void ekgDisable(void);

/**
* \brief Start or restart AppEKG data collection
**/
void ekgEnable(void);

/**
* \brief Provide a name for a heartbeat
* \return 0 on success, -1 on failure
**/
int ekgNameHeartbeat(unsigned int id, char* name);

/**
* \brief Find ID of heartbeat from name
* \return ID value on success, 0 on failure
**/
unsigned int ekgIdOfHeartbeat(char* name);

/**
* \brief Find name of heartbeat from ID
* \return new allocated copy of name on success, 0 on failure
**/
char* ekgNameOfHeartbeat(unsigned int id);

// Rate factor control data
// in-library sources needs to define EKG_EXTERN as empty
#ifndef EKG_EXTERN
#define EKG_EXTERN extern
#endif
EKG_EXTERN unsigned int* _ekgHBEndFlag;
EKG_EXTERN unsigned int* _ekgHBCount;
// track the actual thread id that is hashed to this location, other
// threads that hash here will be ignored
EKG_EXTERN unsigned int* _ekgActualThreadID;
EKG_EXTERN unsigned long (*_ekgThreadId)(void);

#endif // APPEKG_DISABLE

#ifdef __cplusplus
}
#endif

#endif // ifdef APPEKG_H
