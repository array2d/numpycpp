#include <numpy/core.h>
#include <iostream>
#include <vector>

int main() {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    std::cout << "sum:  " << numpy::sum(data, 5) << std::endl;
    std::cout << "mean: " << numpy::mean(data, 5) << std::endl;
    std::cout << "max:  " << numpy::max(data, 5) << std::endl;
    std::cout << "min:  " << numpy::min(data, 5) << std::endl;

    double sq[5];
    numpy::sqrt(data, sq, 5);
    std::cout << "sqrt:";
    for (int i = 0; i < 5; ++i) std::cout << " " << sq[i];
    std::cout << std::endl;

    return 0;
}
