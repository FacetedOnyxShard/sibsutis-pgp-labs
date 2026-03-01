#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

vector<int> a;
vector<int> b;
vector<int> res;
constexpr size_t SIZE_VECTORS = 1 << 20;

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

  for (size_t i = 0; i < SIZE_VECTORS; ++i) {
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
    size_t start = thread_id * (SIZE_VECTORS / num_threads);
    size_t end = (thread_id == num_threads - 1)
                     ? SIZE_VECTORS
                     : (thread_id + 1) * (SIZE_VECTORS / num_threads);

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
  a.resize(SIZE_VECTORS);
  b.resize(SIZE_VECTORS);
  res.resize(SIZE_VECTORS);
}

void init_before_change_method(size_t &a_sum, size_t &b_sum) {
  a_sum = 0;
  b_sum = 0;

  for (size_t i = 0; i < SIZE_VECTORS; ++i) {
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

auto add_parg(
    int threadsInBlock = 256) { // этот код работает, просто не то расширение
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  init_after_change_size();

  int *d_a, *d_b, *d_res;
  size_t vector_byte_size = SIZE_VECTORS * sizeof(int);
  cudaMalloc(&d_a, vector_byte_size);
  cudaMalloc(&d_b, vector_byte_size);
  cudaMalloc(&d_res, vector_byte_size);

  cudaMemcpy(d_a, a.data(), vector_byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), vector_byte_size, cudaMemcpyHostToDevice);

  int blocksCount = (SIZE_VECTORS + threadsInBlock - 1) / threadsInBlock;

  cudaEventRecord(start);
  vector_add<<<blocksCount, threadsInBlock>>>(d_a, d_b, d_res, SIZE_VECTORS);
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

  int center = SIZE_VECTORS / 2 - 1;
  for (int i = center; i < center + proba_size; i++) {
    if (res[i] != a[i] + b[i]) {
      ok = false;
      break;
    }
  }

  int end = SIZE_VECTORS - 1;
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

  printf("%-35s %-35s %-20s\n", "Количество нитей", "Время выполнения",
         "Результат корректен?");

  for (int threadsInBlock = 1; threadsInBlock <= 1024; threadsInBlock <<= 1) {
    if (threadsInBlock != 1 && threadsInBlock < 16)
      continue;

    auto [time_ns, ok] = add_parg(threadsInBlock);
    printf("%-20d %-20f %-20s\n", threadsInBlock, time_ns, (ok ? "YES" : "NO"));
  }
  cout << '\n';

  return 0;
}
