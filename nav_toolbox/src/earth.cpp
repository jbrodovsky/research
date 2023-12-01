#include <cmath>
#include <vector>
#include <array>

// Constants
const double RATE = 7.2921157e-5;
const double G0 = 9.8;
const double R0 = 6378137.0;
const double E2 = 6.6943799901413e-3;
const double GE = 9.7803253359;
const double GP = 9.8321849378;

// Function to convert degrees to radians
/** 
 * @brief Convert degrees to radians
 * @param deg Angle in degrees
 * @return Angle in radians
*/
double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

// Function to compute the principal radii of curvature of Earth ellipsoid
std::array<double, 3> principal_radii(double lat, double alt) {
    double sin_lat = std::sin(deg2rad(lat));
    double cos_lat = std::sqrt(1 - std::pow(sin_lat, 2));

    double x = 1 - E2 * std::pow(sin_lat, 2);
    double re = R0 / std::sqrt(x);
    double rn = re * (1 - E2) / x;

    return {rn + alt, re + alt, (re + alt) * cos_lat};
}

// Function to compute gravity according to a theoretical model
double gravity(double lat, double alt) {
    double sin_lat = std::sin(deg2rad(lat));
    double F = std::sqrt(1 - E2) * GP / GE - 1;
    return (GE * (1 + F * std::pow(sin_lat, 2)) / std::sqrt(1 - E2 * std::pow(sin_lat, 2))
            * (1 - 2 * alt / R0));
}

// Function to compute gravity vector in NED frame
std::vector<double> gravity_n(double lat, double alt) {
    double g = gravity(lat, alt);
    return {0, 0, g};
}

// Function to compute Earth curvature matrix
std::array<std::array<double, 3>, 3> curvature_matrix(double lat, double alt) {
    auto [rn, re, _] = principal_radii(lat, alt);

    std::array<std::array<double, 3>, 3> result = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    result[0][1] = 1 / re;
    result[1][0] = -1 / rn;
    result[2][1] = -result[0][1] * std::tan(deg2rad(lat));

    return result;
}

// Function to compute Earth rate resolved in NED frame
std::array<double, 3> rate_n(double lat) {
    std::array<double, 3> result = {0, 0, 0};
    result[0] = RATE * std::cos(deg2rad(lat));
    result[2] = -RATE * std::sin(deg2rad(lat));
    return result;
}