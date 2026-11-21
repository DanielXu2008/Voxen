#include <iostream>
#include <format>

int main() {
    std::cout << std::format("Hello from {} using C++{}\n", "Voxen", __cplusplus);
}
