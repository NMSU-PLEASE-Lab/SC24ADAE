#
# Build file for AppEKG
#

# LDMS streams support
DO_USE_LDMS_STREAMS = OFF
LDMSDIR = /project/hpcjobquality/tools/INSTALL/OVIS
ifeq ($(DO_USE_LDMS_STREAMS),ON)
CFLAGS = -DUSE_LDMS_STREAMS -I${LDMSDIR}/include
else 
CFLAGS = 
endif

.phony: all doc clean

all: libappekg.a 

libappekg.a: appekg.o
	ar crs libappekg.a appekg.o

appekg.o: appekg.c
	gcc -Wall -c $(CFLAGS) appekg.c

clean:
	rm -rf *.o *.a

doc: Doxyfile appekg.c appekg.h
	doxygen Doxyfile

