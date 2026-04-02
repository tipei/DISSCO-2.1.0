#ifndef DISSCO_MPI_WRAPPER_H
#define DISSCO_MPI_WRAPPER_H

#include <cstdlib>
#include <iostream>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace dissco_mpi {

inline void finalizeAtExit() {
#ifdef USE_MPI
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Finalize();
  }
#endif
}

inline void ensureInitialized() {
#ifdef USE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    return;
  }

  int provided = 0;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
  if (provided < MPI_THREAD_SERIALIZED) {
    std::cerr << "MPI runtime does not provide MPI_THREAD_SERIALIZED support."
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
  }

  static bool registered = false;
  if (!registered) {
    std::atexit(finalizeAtExit);
    registered = true;
  }
#endif
}

inline bool isActive() {
#ifdef USE_MPI
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return false;
  }
  MPI_Finalized(&finalized);
  return finalized == 0;
#else
  return false;
#endif
}

inline int rank() {
#ifdef USE_MPI
  if (!isActive()) {
    return 0;
  }
  int mpiRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  return mpiRank;
#else
  return 0;
#endif
}

inline int size() {
#ifdef USE_MPI
  if (!isActive()) {
    return 1;
  }
  int mpiSize = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  return mpiSize;
#else
  return 1;
#endif
}

inline bool isRoot() {
  return rank() == 0;
}

inline void barrier() {
#ifdef USE_MPI
  if (isActive()) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

inline void broadcastInt(int& value) {
#ifdef USE_MPI
  if (isActive()) {
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
#else
  (void) value;
#endif
}

inline void broadcastString(std::string& value) {
#ifdef USE_MPI
  if (!isActive()) {
    return;
  }

  int length = static_cast<int>(value.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  value.resize(length);

  if (length > 0) {
    MPI_Bcast(&value[0], length, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#else
  (void) value;
#endif
}

inline int localRenderThreads(int configuredThreads) {
  if (size() > 1) {
    return 1;
  }
  return configuredThreads;
}

}  // namespace dissco_mpi

#endif
