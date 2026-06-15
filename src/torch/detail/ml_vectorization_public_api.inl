// Public API

at::Tensor ml_persistence_image(
    const at::Tensor& diagram,
    int64_t resolution_birth,
    int64_t resolution_death,
    double sigma,
    double birth_min,
    double birth_max,
    double death_min,
    double death_max,
    const std::string& weight_fn
) {
    return persistence_image_impl(diagram, resolution_birth, resolution_death,
                                   sigma, birth_min, birth_max, death_min, death_max, weight_fn);
}

at::Tensor ml_persistence_landscape(
    const at::Tensor& diagram,
    int64_t k,
    int64_t num_samples,
    double x_min,
    double x_max
) {
    return persistence_landscape_impl(diagram, k, num_samples, x_min, x_max);
}

at::Tensor ml_persistence_silhouette(
    const at::Tensor& diagram,
    int64_t num_samples,
    double x_min,
    double x_max,
    const std::string& weight_fn
) {
    return persistence_silhouette_impl(diagram, num_samples, x_min, x_max, weight_fn);
}

at::Tensor ml_heat_kernel_signature(
    const at::Tensor& diagram,
    int64_t num_samples,
    double sigma,
    const at::Tensor& t_values
) {
    return heat_kernel_signature_impl(diagram, num_samples, sigma, t_values);
}

at::Tensor ml_birth_death_curve(
    const at::Tensor& diagram,
    int64_t num_bins,
    const std::string& statistic
) {
    return birth_death_curve_impl(diagram, num_bins, statistic);
}
