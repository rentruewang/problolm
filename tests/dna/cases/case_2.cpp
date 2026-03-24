#include <iostream>
#include <vector>

/// Here is the docstring for the function.
int compute_sum(const std::vector<int> &data) {
  int sum = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] % 2 == 0) {
      sum += data[i] / 2;
    } else {
      sum += data[i] * 2;
    }
  }
  return sum;
}

int main() {
  std::vector<int> numbers = {3, 7, 2, 9, 4, 6};
  std::cout << "Result: " << compute_sum(numbers) << std::endl;
  return 0;
}
