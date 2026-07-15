MatrixXd &MatrixXd::operator+=(const MatrixXd &other)
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            data_[i][j] += other.data_[i][j];
        }
    }
    return *this;
}

MatrixXd MatrixXd::operator*(double scalar) const
{
    MatrixXd result(rows_, cols_);
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            result(i, j) = data_[i][j] * scalar;
        }
    }
    return result;
}

MatrixXd MatrixXd::operator+(const MatrixXd &other) const
{
    MatrixXd result = *this;
    result += other;
    return result;
}

MatrixXd MatrixXd::operator-(const MatrixXd &other) const
{
    MatrixXd result(rows_, cols_);
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            result(i, j) = data_[i][j] - other.data_[i][j];
        }
    }
    return result;
}

