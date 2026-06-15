namespace {

template <typename Exception>
void set_registered_exception(const py::exception<Exception>& exception_type, const char* message) {
    py::set_error(exception_type.ptr(), message);
}

}  // namespace

void register_exception_translators(py::module& m) {
    // Register NerveError and subclasses
    static py::exception<nerve::errors::NerveError> exc_nerve(m, "NerveError");
    static py::exception<nerve::errors::ShapeMismatchError> exc_shape(m, "ShapeMismatchError");
    static py::exception<nerve::errors::DimensionError> exc_dim(m, "DimensionError");
    static py::exception<nerve::errors::TypeError> exc_type(m, "TypeMismatchError");
    static py::exception<nerve::errors::InvalidSimplexError> exc_simplex(m, "InvalidSimplexError");
    static py::exception<nerve::errors::MatrixStructureError> exc_matrix(m, "MatrixStructureError");
    static py::exception<nerve::errors::GPUError> exc_gpu(m, "GPUError");
    static py::exception<nerve::errors::GPUMemoryError> exc_gpu_mem(m, "GPUMemoryError");
    static py::exception<nerve::errors::GPULaunchError> exc_gpu_launch(m, "GPULaunchError");
    static py::exception<nerve::errors::MemoryError> exc_mem(m, "MemoryError");
    static py::exception<nerve::errors::OutOfMemoryError> exc_oom(m, "OutOfMemoryError");
    static py::exception<nerve::errors::AllocationError> exc_alloc(m, "AllocationError");
    static py::exception<nerve::errors::NumericalError> exc_num(m, "NumericalError");
    static py::exception<nerve::errors::ConvergenceError> exc_conv(m, "ConvergenceError");
    static py::exception<nerve::errors::PrecisionError> exc_prec(m, "PrecisionError");
    static py::exception<nerve::errors::NumericalInstabilityError> exc_inst(m, "NumericalInstabilityError");
    static py::exception<nerve::errors::InvalidArgumentError> exc_arg(m, "InvalidArgumentError");
    static py::exception<nerve::errors::BudgetExceededError> exc_budget(m, "BudgetExceededError");
    static py::exception<nerve::errors::IOError> exc_io(m, "IOError");
    static py::exception<nerve::errors::DeterminismError> exc_det(m, "DeterminismError");
    static py::exception<nerve::errors::NUMAError> exc_numa(m, "NUMAError");
    static py::exception<nerve::errors::PersistenceError> exc_pers(m, "PersistenceError");
    static py::exception<nerve::errors::BettiError> exc_betti(m, "BettiError");

    // Register translators - order matters (most specific first)
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const nerve::errors::BettiError& e) {
            set_registered_exception(exc_betti, e.what());
        } catch (const nerve::errors::PersistenceError& e) {
            set_registered_exception(exc_pers, e.what());
        } catch (const nerve::errors::NUMAError& e) {
            set_registered_exception(exc_numa, e.what());
        } catch (const nerve::errors::DeterminismError& e) {
            set_registered_exception(exc_det, e.what());
        } catch (const nerve::errors::IOError& e) {
            set_registered_exception(exc_io, e.what());
        } catch (const nerve::errors::BudgetExceededError& e) {
            set_registered_exception(exc_budget, e.what());
        } catch (const nerve::errors::InvalidArgumentError& e) {
            set_registered_exception(exc_arg, e.what());
        } catch (const nerve::errors::NumericalInstabilityError& e) {
            set_registered_exception(exc_inst, e.what());
        } catch (const nerve::errors::PrecisionError& e) {
            set_registered_exception(exc_prec, e.what());
        } catch (const nerve::errors::ConvergenceError& e) {
            set_registered_exception(exc_conv, e.what());
        } catch (const nerve::errors::NumericalError& e) {
            set_registered_exception(exc_num, e.what());
        } catch (const nerve::errors::AllocationError& e) {
            set_registered_exception(exc_alloc, e.what());
        } catch (const nerve::errors::OutOfMemoryError& e) {
            set_registered_exception(exc_oom, e.what());
        } catch (const nerve::errors::MemoryError& e) {
            set_registered_exception(exc_mem, e.what());
        } catch (const nerve::errors::GPULaunchError& e) {
            set_registered_exception(exc_gpu_launch, e.what());
        } catch (const nerve::errors::GPUMemoryError& e) {
            set_registered_exception(exc_gpu_mem, e.what());
        } catch (const nerve::errors::GPUError& e) {
            set_registered_exception(exc_gpu, e.what());
        } catch (const nerve::errors::MatrixStructureError& e) {
            set_registered_exception(exc_matrix, e.what());
        } catch (const nerve::errors::InvalidSimplexError& e) {
            set_registered_exception(exc_simplex, e.what());
        } catch (const nerve::errors::TypeError& e) {
            set_registered_exception(exc_type, e.what());
        } catch (const nerve::errors::DimensionError& e) {
            set_registered_exception(exc_dim, e.what());
        } catch (const nerve::errors::ShapeMismatchError& e) {
            set_registered_exception(exc_shape, e.what());
        } catch (const nerve::errors::NerveError& e) {
            set_registered_exception(exc_nerve, e.what());
        }
    });
}

// Module definition
