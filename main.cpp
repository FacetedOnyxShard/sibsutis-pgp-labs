#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

class VectorAddition {
  vector<int> a;
  vector<int> b;
  vector<int> res;
  size_t a_checksum;
  size_t b_checksum;
  size_t res_checksum;
  size_t n;

  void init() {
    a.resize(n);
    b.resize(n);
    res.resize(n);

    a_checksum = 0;
    b_checksum = 0;
    res_checksum = 0;

    for (int i = 0; i < n; ++i) {
      int a_val = rand() % 100;
      int b_val = rand() % 100;

      a[i] = a_val;
      b[i] = b_val;

      a_checksum += a_val;
      b_checksum += b_val;
    }
  }

public:
  VectorAddition(const size_t n = 10) {
    this->n = n;
    init();
  }

  void DoSeq() {
    transform(a.begin(), a.end(), b.begin(), res.begin(),
              [](int a, int b) { return a + b; });
  }

  void DoParallel() {}

  void show() {
    for (int i = 0; i < n; ++i) {
      cout << a[i] << " + " << b[i] << " = " << res[i] << '\n';
    }
  }

  void show_first(const size_t n) {
    if (n > this->n)
      return;

    for (int i = 0; i < n; ++i) {
      printf("%2d + %2d = %3d\n", a[i], b[i], res[i]);
    }
  }

  bool check() {
    res_checksum = accumulate(res.begin(), res.end(), 0);

    if (res_checksum == a_checksum + b_checksum)
      return true;
    return false;
  }
};

template <typename Callback> void rep(const size_t n, Callback func) {
  for (size_t i = 0; i < n; ++i) {
    func();
  }
}

void print_line(const char ch, const size_t n) {
  rep(n, [&ch]() { printf("%c", ch); });
  printf("\n");
}

int main(void) {
  // size_t max_n_for_tests = 10000000;
  // size_t opt_ls = 80;
  // srand(time(NULL));

  // for (size_t n = 1000; n <= max_n_for_tests; n *= 10) {
  //   VectorAddition va(n);

  //   auto start = high_resolution_clock::now();
  //   va.DoSeq();
  //   auto end = high_resolution_clock::now();
  //   auto duration = duration_cast<nanoseconds>(end - start);

  //   print_line('=', opt_ls);
  //   cout << "Checksum is correct: " << (va.check() ? "YES" : "NO") << '\n';
  //   cout << "Execution time (nanoseconds): " << duration.count() << '\n';
  //   va.show_first(10);
  // }
  // print_line('=', opt_ls);

  int num_threads = thread::hardware_concurrency();
  cout << "Количество доступных потоков на устройстве: " << num_threads << '\n';

  return 0;
}
