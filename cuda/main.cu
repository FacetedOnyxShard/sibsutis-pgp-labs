// тольк работающая gpu
// #include <stdio.h>
// #include <bits/stdc++.h>
// #include <stdlib.h>
// #include <time.h>

// using namespace std;

// __global__ void vector_add(int *a, int *b, int *res, int vectrs_size) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x ;
//   if (i < vectrs_size) res[i] = a[i] + b[i];
// }

// vector<int> a;
// vector<int> b;
// vector<int> res;
// size_t size_vectors;


// void init_after_change_size() {
//   a.resize(size_vectors);
//   b.resize(size_vectors);
//   res.resize(size_vectors);
// }

// void init_before_change_method(size_t &a_sum, size_t &b_sum) {
//   a_sum = 0;
//   b_sum = 0;

//   for (size_t i = 0; i < size_vectors; ++i) {
//     a[i] = rand() % 100;
//     b[i] = rand() % 100;
//     a_sum += a[i];
//     b_sum += b[i];
//   }
// }


// int main(void) {
//   srand(time(NULL));
//   size_vectors = 1000;

//   size_t a_sum = 0;
//   size_t b_sum = 0;

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   init_after_change_size();
//   init_before_change_method(a_sum, b_sum);

//   int *d_a, *d_b, *d_res;
//   size_t vector_byte_size = size_vectors * sizeof(int);
//   cudaMalloc(&d_a, vector_byte_size);
//   cudaMalloc(&d_b, vector_byte_size);
//   cudaMalloc(&d_res, vector_byte_size);

//   cudaMemcpy(d_a, a.data(), vector_byte_size, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_b, b.data(), vector_byte_size, cudaMemcpyHostToDevice);

//   int threadsInBlock = 256;
//   int blocksCount = (size_vectors + threadsInBlock - 1) / threadsInBlock;

//   cudaEventRecord(start);
//   vector_add<<< blocksCount, threadsInBlock >>>(d_a, d_b, d_res, size_vectors);
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);

//   float time_ms;
//   cudaEventElapsedTime(&time_ms, start, stop);
//   float time_ns = time_ms * 1000000.0f; 

//   cudaMemcpy(res.data(), d_res, vector_byte_size, cudaMemcpyDeviceToHost);


//   bool ok = true;
//   int proba_size = 10;
//   for (int i = 0; i < proba_size; i++) {
//     if (res[i] != a[i] + b[i]) {
//         ok = false;
//         break;
//     }
//   }

//   int center = size_vectors / 2 - 1;
//   for (int i = center; i < center + proba_size; i++) {
//     if (res[i] != a[i] + b[i]) {
//         ok = false;
//         break;
//     }
//   }
  
//   int end = size_vectors - 1;
//   for (int i = end; i > end - proba_size; --i) {
//     if (res[i] != a[i] + b[i]) {
//         ok = false;
//         break;
//     }
//   }


//   cudaFree(d_a);
//   cudaFree(d_b);
//   cudaFree(d_res);


//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);

//   return 0;
// }

// вся программа
#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

vector<int> a;
vector<int> b;
vector<int> res;
size_t size_vectors;

template <typename Func, typename... Args>
auto measure_time(Func func, Args... args) {
  auto start = high_resolution_clock::now();

  auto res = (func)((args)...);

  auto end = high_resolution_clock::now();
  auto dur = duration_cast<nanoseconds>(end - start);

  return make_pair(dur.count(), res);
}

long long add_seq() {
  long long sum = 0;

  for (size_t i = 0; i < size_vectors; ++i) {
    res[i] = a[i] + b[i];
    sum += res[i];
  }

  return sum;
}

long long add_parc() {
  long long total_sum = 0;
  const size_t num_threads = thread::hardware_concurrency();
  vector<thread> threads;
  vector<long long> partial_sums(num_threads, 0);

  auto worker = [&](size_t thread_id) {
    long long local_sum = 0;
    size_t start = thread_id * (size_vectors / num_threads);
    size_t end = (thread_id == num_threads - 1)
                     ? size_vectors
                     : (thread_id + 1) * (size_vectors / num_threads);

    for (size_t i = start; i < end; ++i) {
      res[i] = a[i] + b[i];
      local_sum += res[i];
    }

    partial_sums[thread_id] = local_sum;
  };

  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker, i);
  }

  for (auto &t : threads) {
    t.join();
  }

  for (auto &partial : partial_sums) {
    total_sum += partial;
  }

  return total_sum;
}

void init_after_change_size() {
  a.resize(size_vectors);
  b.resize(size_vectors);
  res.resize(size_vectors);
}

void init_before_change_method(size_t &a_sum, size_t &b_sum) {
  a_sum = 0;
  b_sum = 0;

  for (size_t i = 0; i < size_vectors; ++i) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    a_sum += a[i];
    b_sum += b[i];
  }
}

__global__ void vector_add(int *a, int *b, int *res, int vectrs_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < vectrs_size)
    res[i] = a[i] + b[i];
}

auto add_parg() { // этот код работает, просто не то расширение
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  init_after_change_size();

  int *d_a, *d_b, *d_res;
  size_t vector_byte_size = size_vectors * sizeof(int);
  cudaMalloc(&d_a, vector_byte_size);
  cudaMalloc(&d_b, vector_byte_size);
  cudaMalloc(&d_res, vector_byte_size);

  cudaMemcpy(d_a, a.data(), vector_byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), vector_byte_size, cudaMemcpyHostToDevice);

  int threadsInBlock = 256;
  int blocksCount = (size_vectors + threadsInBlock - 1) / threadsInBlock;

  cudaEventRecord(start);
  vector_add<<<blocksCount, threadsInBlock>>>(d_a, d_b, d_res, size_vectors);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);
  float time_ns = time_ms * 1000000.0f;

  cudaMemcpy(res.data(), d_res, vector_byte_size, cudaMemcpyDeviceToHost);

  bool ok = true;
  int proba_size = 10;
  for (int i = 0; i < proba_size; i++) {
    if (res[i] != a[i] + b[i]) {
      ok = false;
      break;
    }
  }

  int center = size_vectors / 2 - 1;
  for (int i = center; i < center + proba_size; i++) {
    if (res[i] != a[i] + b[i]) {
      ok = false;
      break;
    }
  }

  int end = size_vectors - 1;
  for (int i = end; i > end - proba_size; --i) {
    if (res[i] != a[i] + b[i]) {
      ok = false;
      break;
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_res);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return make_pair(time_ns, ok);
}

void print_line(char ch, int n) {
  for (int i = 0; i < n; ++i)
    printf("%c", ch);

  printf("\n");
}

int main() {
  srand(time(NULL));

  size_t a_sum = 0;
  size_t b_sum = 0;

  printf("%-15s %-12s %-12s %-15s %-15s %-15s %-15s\n", "Размер", "Время(пос)",
         "Рез(пос)", "Время(пар cpu)", "Рез(пар cpu)", "Время(пар gpu)",
         "Рез(пар gpu)");
  print_line('=', 100);

  for (size_vectors = 1000; size_vectors <= 10000000; size_vectors *= 100) {
    init_after_change_size();
    init_before_change_method(a_sum, b_sum);

    auto [seq_time, seq_res] = measure_time(add_seq);
    bool seq_ok = ((long long)(a_sum + b_sum) == seq_res);

    init_before_change_method(a_sum, b_sum);
    auto [parc_time, parc_res] = measure_time(add_parc);
    bool parc_ok = ((long long)(a_sum + b_sum) == parc_res);

    auto [parg_time, parg_ok] = add_parg();

    printf("%-15zu %-12lld %-12s %-15lld %-15s %-15.0f %-15s\n", size_vectors,
           seq_time, (seq_ok ? "OK" : "FAIL"), parc_time,
           (parc_ok ? "OK" : "FAIL"), parg_time, (parg_ok ? "OK" : "FAIL"));
  }
  print_line('=', 100);

  return 0;
}
