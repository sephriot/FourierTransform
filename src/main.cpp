#include <complex>
#include <iostream>
#include <fstream>
#include <valarray>
#include <vector>

namespace {
const double PI = 3.141592653589793238;
}

using Complex = std::complex<double>;
using CArray = std::valarray<Complex>;
using CVector = std::vector<Complex>;


CVector DFT(const CVector& x)
{
    CVector ret;
    const size_t N = x.size();
    for (size_t i = 0; i < N; ++i) {
        Complex sum(0.0, 0.0);

        for (size_t j = 0; j < N; ++j)
        {
            // std::exp(Complex(0.0, 2 * PI * j * i / N));
            sum += x[j] * std::polar(1.0, 2 * PI * j * i / N);
        }

        ret.push_back(abs(sum));
    }

    return std::move(ret);
}

// Cooley–Tukey FFT (in-place, divide-and-conquer)
void FFT(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) {
        return;
    }

    // divide
    CArray even = x[std::slice(0, N / 2, 2)];
    CArray odd = x[std::slice(1, N / 2, 2)];

    // conquer
    FFT(even);
    FFT(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, 2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// inverse fft (in-place)
void IFFT(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    FFT(x);

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
}

int main()
{
    std::ifstream input("dane.txt");
    std::vector<Complex> dataV;
    double Re;
    double Im;
    while (!input.eof()) {
        input >> Re >> Im;
        std::cout << "(" << Re << "," << Im << ")" << std::endl;
        dataV.emplace_back(Re, Im);
    }

    input.close();


    std::cout << std::endl << "=== DFT ===" << std::endl;
    for(const auto& it : DFT(dataV)) {
        std::cout << it << std::endl;
    }



    CArray data(dataV.data(), dataV.size());
    // forward fft
    FFT(data);
    std::cout << std::endl << "=== FFT ===" << std::endl;
    for (const auto& it : data) {
        std::cout << it << std::endl;
    }

    // inverse fft
    IFFT(data);
    std::cout << std::endl << "=== IFFT ===" << std::endl;
    for (const auto& it : data) {
        std::cout << it << std::endl;
    }
    return 0;
}