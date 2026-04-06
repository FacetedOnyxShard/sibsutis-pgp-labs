#include <bits/stdc++.h>
#include <cstddef>
#include <cstdlib>

using namespace std;

int main() {
  srand(time(NULL));

  constexpr size_t matrix_size = 2;
  vector<vector<double>> a(matrix_size, vector<double>(matrix_size));
  vector<vector<double>> b(matrix_size, vector<double>(matrix_size));
  vector<vector<double>> c(matrix_size, vector<double>(matrix_size));

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      a[i][j] = rand() % 5 + 1;
      b[i][j] = rand() % 5 + 1;
      c[i][j] = 0;
    }
  }

  cout << "Значения a:\n";
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      cout << a[i][j] << ' ';
    }
    cout << '\n';
  }
  cout << "Значения b:\n";
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      cout << b[i][j] << ' ';
    }
    cout << '\n';
  }
  cout << "Значения c:\n";
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      cout << c[i][j] << ' ';
    }
    cout << '\n';
  }

  // for (size_t i = 0; i < matrix_size; ++i) {
  //   for (size_t j = 0; j < matrix_size; ++j) {
  //     double sum = 0;
  //     for (size_t k = 0; k < matrix_size; ++k) {
  //       sum += a[i][j + k] * b[i + k][j];
  //     }
  //     c[i][j] = sum;
  //   }
  // }

  cout << "Результат:\n";
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      cout << c[i][j] << ' ';
    }
    cout << '\n';
  }

  return 0;
}