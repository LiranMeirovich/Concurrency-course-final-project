# Parallel String Alignment with CUDA, MPI, and OpenMP

## 🚀 Overview
A high-performance computing solution that combines CUDA GPU acceleration, MPI distributed computing, and OpenMP multi-threading to solve complex string alignment problems. This project demonstrates expertise in parallel programming across multiple paradigms and architectures.

## 🔑 Key Features
- **Hybrid Parallelization**: Leverages three major parallel computing paradigms:
  - CUDA for GPU acceleration of intensive string comparisons
  - MPI for distributed computing across multiple nodes
  - OpenMP for shared-memory parallelization
- **Optimized Performance**:
  - Custom CUDA kernels for parallel string alignment
  - Efficient memory management with CUDA shared memory
  - Load balancing across MPI processes
  - Reduction operations optimized for GPU architecture
- **Advanced Algorithms**:
  - Parallel prefix scan implementation
  - Custom reduction operations for string alignment scores
  - Mutation-based string alignment scoring

## 🏗️ Architecture

### Core Components
- `ParallelProject.c`: Main orchestrator implementing MPI and OpenMP parallelization
- `cudaFunctions.cu`: CUDA kernel implementations for GPU-accelerated string alignment
- `scan_strcmp.cu`: Specialized CUDA implementation for parallel string comparison

### Technical Specifications
- CUDA Block Dimension: 1024 threads (optimized for modern GPU architectures)
- Dynamic workload distribution between CPU and GPU
- Custom MPI data types for efficient communication
- Shared memory utilization for optimal GPU performance

### Hybrid Parallelization Strategy
- GPU handles compute-intensive string alignments
- MPI processes distribute workload across nodes
- OpenMP manages thread-level parallelism within each node

## 🛠️ Build & Usage

### Prerequisites
- CUDA Toolkit (11.0 or higher)
- OpenMPI implementation
- OpenMP-compatible compiler
- GCC with CUDA support

### Compilation
```bash
# Compile CUDA components
nvcc -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o

# Link everything together
gcc -o mpiCudaOpenMP main.o cudaFunctions.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
```

## 🎯 Performance Considerations
- Optimal for large-scale string alignment problems
- Scalable across multiple GPU nodes
- Efficient memory utilization through shared memory and coalesced access patterns
- Load balancing achieved through dynamic work distribution

## 💡 Technical Achievements
- Implementation of parallel prefix scan on GPU
- Custom MPI reduction operations for complex data types
- Efficient hybrid memory management across distributed systems
- Advanced string mutation algorithms for alignment scoring

## 🔬 Learning Outcomes
This project demonstrates proficiency in:
- Parallel programming paradigms
- High-performance computing
- GPU architecture and optimization
- Distributed systems
- Algorithm design and optimization
- Memory management in heterogeneous systems
