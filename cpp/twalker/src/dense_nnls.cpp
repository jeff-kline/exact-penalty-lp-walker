#include "dense_nnls.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace twalker {
namespace {

class MaintainedNnls {
 public:
  MaintainedNnls(const std::vector<double> &matrix, int rows, int columns,
                 const std::vector<double> &rhs)
      : a_(matrix), rows_(rows), columns_(columns), rhs_(rhs),
        x_(columns, 0.0), candidate_(columns, 0.0),
        residual_(rows, 0.0), gradient_(columns, 0.0),
        column_(rows, 0.0), passive_(columns, 0),
        q_(static_cast<std::size_t>(rows) * columns, 0.0),
        r_(static_cast<std::size_t>(columns) * columns, 0.0),
        qtb_(columns, 0.0), coefficients_(columns, 0.0) {
    order_.reserve(columns);
  }

  DenseNnlsResult solve(int max_iterations) {
    DenseNnlsResult result;
    if (rows_ <= 0 || columns_ <= 0
        || a_.size() != static_cast<std::size_t>(rows_) * columns_
        || rhs_.size() != static_cast<std::size_t>(rows_))
      return result;
    double matrix_one_norm = 0.0;
    for (int column = 0; column < columns_; ++column) {
      double sum = 0.0;
      for (int row = 0; row < rows_; ++row)
        sum += std::abs(a_[row + static_cast<std::size_t>(rows_) * column]);
      matrix_one_norm = std::max(matrix_one_norm, sum);
    }
    const double tolerance = 10.0 * std::max(rows_, columns_)
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, matrix_one_norm);

    int iterations = 0;
    while (true) {
      ++outer_iterations_;
      multiply(x_, residual_);
      for (int row = 0; row < rows_; ++row)
        residual_[row] = rhs_[row] - residual_[row];
      multiply_transpose(residual_, gradient_);
      int entering = -1;
      double best = tolerance;
      for (int column = 0; column < columns_; ++column)
        if (!passive_[column] && gradient_[column] > best) {
          best = gradient_[column];
          entering = column;
        }
      if (entering < 0 || static_cast<int>(order_.size()) >= rows_) break;
      if (!append(entering)) {
        passive_[entering] = 1;
        continue;
      }

      while (true) {
        if (++iterations > max_iterations) return finish(false);
        ++inner_iterations_;
        if (!solve_passive()) return finish(false);
        double minimum = 0.0;
        for (int column : order_)
          minimum = std::min(minimum, candidate_[column]);
        if (minimum > 0.0) {
          for (int column : order_) x_[column] = candidate_[column];
          break;
        }
        double alpha = 1.0;
        for (int column : order_) {
          if (candidate_[column] > 0.0) continue;
          const double denominator = x_[column] - candidate_[column];
          if (denominator > 0.0)
            alpha = std::min(alpha, x_[column] / denominator);
        }
        for (int column : order_)
          x_[column] += alpha * (candidate_[column] - x_[column]);
        bool removed = false;
        for (int index = static_cast<int>(order_.size()) - 1;
             index >= 0; --index) {
          const int column = order_[index];
          if (std::abs(x_[column]) <= 1e-14) {
            x_[column] = 0.0;
            passive_[column] = 0;
            erase(index);
            removed = true;
          }
        }
        if (!removed) break;
      }
    }
    return finish(true);
  }

 private:
  const std::vector<double> &a_;
  int rows_, columns_;
  const std::vector<double> &rhs_;
  std::vector<double> x_, candidate_, residual_, gradient_, column_;
  std::vector<int> order_;
  std::vector<std::uint8_t> passive_;
  std::vector<double> q_, r_, qtb_, coefficients_;
  std::int64_t inner_iterations_ = 0, outer_iterations_ = 0;

  void multiply(const std::vector<double> &x, std::vector<double> &out) const {
    std::fill(out.begin(), out.end(), 0.0);
    for (int column = 0; column < columns_; ++column) {
      if (x[column] == 0.0) continue;
      const double *source = &a_[static_cast<std::size_t>(rows_) * column];
      for (int row = 0; row < rows_; ++row)
        out[row] += source[row] * x[column];
    }
  }

  void multiply_transpose(const std::vector<double> &x,
                          std::vector<double> &out) const {
    for (int column = 0; column < columns_; ++column) {
      const double *source = &a_[static_cast<std::size_t>(rows_) * column];
      double sum = 0.0;
      for (int row = 0; row < rows_; ++row) sum += source[row] * x[row];
      out[column] = sum;
    }
  }

  bool append(int entering) {
    const int count = static_cast<int>(order_.size());
    const double *source = &a_[static_cast<std::size_t>(rows_) * entering];
    std::copy_n(source, rows_, column_.begin());
    double original_norm2 = 0.0;
    for (double value : column_) original_norm2 += value * value;
    const double original_norm = std::sqrt(original_norm2);
    if (!(original_norm > 0.0)) return false;
    std::fill(coefficients_.begin(), coefficients_.begin() + count, 0.0);
    for (int pass = 0; pass < 2; ++pass) {
      for (int index = 0; index < count; ++index) {
        const double *basis = &q_[static_cast<std::size_t>(rows_) * index];
        double product = 0.0;
        for (int row = 0; row < rows_; ++row)
          product += basis[row] * column_[row];
        coefficients_[index] += product;
        for (int row = 0; row < rows_; ++row)
          column_[row] -= product * basis[row];
      }
    }
    double norm2 = 0.0;
    for (double value : column_) norm2 += value * value;
    const double norm = std::sqrt(norm2);
    if (!(norm > 1e-12 * original_norm)) return false;
    double *new_basis = &q_[static_cast<std::size_t>(rows_) * count];
    for (int row = 0; row < rows_; ++row)
      new_basis[row] = column_[row] / norm;
    for (int index = 0; index < count; ++index)
      r_[index + static_cast<std::size_t>(columns_) * count] =
          coefficients_[index];
    r_[count + static_cast<std::size_t>(columns_) * count] = norm;
    double projected_rhs = 0.0;
    for (int row = 0; row < rows_; ++row)
      projected_rhs += new_basis[row] * rhs_[row];
    qtb_[count] = projected_rhs;
    order_.push_back(entering);
    passive_[entering] = 1;
    return true;
  }

  void erase(int removed) {
    const int old_count = static_cast<int>(order_.size());
    for (int column = removed + 1; column < old_count; ++column)
      for (int row = 0; row <= column; ++row)
        r_[row + static_cast<std::size_t>(columns_) * (column - 1)] =
            r_[row + static_cast<std::size_t>(columns_) * column];
    const int new_count = old_count - 1;
    for (int index = removed; index < new_count; ++index) {
      const double first =
          r_[index + static_cast<std::size_t>(columns_) * index];
      const double second =
          r_[index + 1 + static_cast<std::size_t>(columns_) * index];
      if (second == 0.0) continue;
      const double radius = std::hypot(first, second);
      const double cosine = first / radius, sine = second / radius;
      for (int column = index; column < new_count; ++column) {
        double *values = &r_[static_cast<std::size_t>(columns_) * column];
        const double top = values[index], bottom = values[index + 1];
        values[index] = cosine * top + sine * bottom;
        values[index + 1] = -sine * top + cosine * bottom;
      }
      double *q1 = &q_[static_cast<std::size_t>(rows_) * index];
      double *q2 = &q_[static_cast<std::size_t>(rows_) * (index + 1)];
      for (int row = 0; row < rows_; ++row) {
        const double left = q1[row], right = q2[row];
        q1[row] = cosine * left + sine * right;
        q2[row] = -sine * left + cosine * right;
      }
      const double top = qtb_[index], bottom = qtb_[index + 1];
      qtb_[index] = cosine * top + sine * bottom;
      qtb_[index + 1] = -sine * top + cosine * bottom;
    }
    order_.erase(order_.begin() + removed);
  }

  bool solve_passive() {
    std::fill(candidate_.begin(), candidate_.end(), 0.0);
    const int count = static_cast<int>(order_.size());
    for (int index = count - 1; index >= 0; --index) {
      double value = qtb_[index];
      for (int later = index + 1; later < count; ++later)
        value -= r_[index + static_cast<std::size_t>(columns_) * later]
                 * candidate_[order_[later]];
      const double diagonal =
          r_[index + static_cast<std::size_t>(columns_) * index];
      if (std::abs(diagonal) < 1e-300) return false;
      candidate_[order_[index]] = value / diagonal;
    }
    return true;
  }

  DenseNnlsResult finish(bool converged) {
    DenseNnlsResult result;
    result.converged = converged;
    result.x = x_;
    result.inner_iterations = inner_iterations_;
    result.outer_iterations = outer_iterations_;
    multiply(x_, residual_);
    for (int row = 0; row < rows_; ++row)
      residual_[row] -= rhs_[row];
    multiply_transpose(residual_, gradient_);
    double scale = 1.0;
    for (double value : gradient_) scale = std::max(scale, std::abs(value));
    for (int column = 0; column < columns_; ++column) {
      const double violation = x_[column] > 1e-12
          ? std::abs(gradient_[column]) : std::max(0.0, -gradient_[column]);
      result.kkt_error = std::max(result.kkt_error, violation / scale);
    }
    return result;
  }
};

}  // namespace

DenseNnlsResult dense_nnls(const std::vector<double> &matrix, int rows,
                           int columns, const std::vector<double> &rhs,
                           int max_iterations) {
  return MaintainedNnls(matrix, rows, columns, rhs).solve(max_iterations);
}

}  // namespace twalker
