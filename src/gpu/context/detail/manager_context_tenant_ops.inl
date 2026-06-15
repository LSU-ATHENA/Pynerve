MultiTenantManager::MultiTenantManager(GreenContextManager& manager)
    : contextManager_(manager) {
}

WorkloadHandle MultiTenantManager::submitWorkload(
    const WorkloadDescription& desc,
    GreenContextId preferredContext
) {
    WorkloadHandle handle;
    handle.id = nextWorkloadId_++;
    handle.status = WorkloadStatus::kPending;
    if (preferredContext > 0) {
        auto metrics = contextManager_.getContextMetrics(preferredContext);
        if (metrics.active && metrics.memoryUsed < metrics.memoryLimit &&
            desc.memoryRequirement <= metrics.memoryLimit - metrics.memoryUsed) {
            handle.assignedContext = preferredContext;
        }
    }

    if (handle.assignedContext == 0) {
        handle.assignedContext = findBestContext(desc);
    }

    if (handle.assignedContext == 0) {
        const size_t headroom =
            desc.memoryRequirement > std::numeric_limits<size_t>::max() / 2
                ? std::numeric_limits<size_t>::max()
                : desc.memoryRequirement * 2;
        auto ctx = contextManager_.createContext(
            desc.computeFraction,
            headroom,
            desc.priority
        );
        handle.assignedContext = ctx.id;
    }

    workloads_[handle.id] = desc;
    handle.status = WorkloadStatus::kRunning;
    handles_[handle.id] = handle;

    return handle;
}

void MultiTenantManager::cancelWorkload(WorkloadHandle handle) {
    auto it = workloads_.find(handle.id);
    if (it != workloads_.end()) {
        it->second.cancelled = true;
        if (auto handleIt = handles_.find(handle.id); handleIt != handles_.end()) {
            handleIt->second.status = WorkloadStatus::kCancelled;
        }
    }
}

void MultiTenantManager::prioritizeWorkload(WorkloadHandle handle, int priority) {
    auto it = workloads_.find(handle.id);
    if (it != workloads_.end()) {
        it->second.priority = priority;
        contextManager_.setContextPriority(handle.assignedContext, priority);
    }
}

WorkloadStatus MultiTenantManager::getWorkloadStatus(WorkloadId id) const {
    auto handleIt = handles_.find(id);
    if (handleIt == handles_.end()) {
        return WorkloadStatus::kError;
    }
    return handleIt->second.status;
}

GreenContextId MultiTenantManager::findBestContext(const WorkloadDescription& desc) {
    auto activeContexts = contextManager_.getActiveContexts();

    GreenContextId bestContext = 0;
    float bestScore = -1.0f;

    for (auto ctxId : activeContexts) {
        auto metrics = contextManager_.getContextMetrics(ctxId);

        if (metrics.memoryUsed >= metrics.memoryLimit ||
            desc.memoryRequirement > metrics.memoryLimit - metrics.memoryUsed) {
            continue;
        }

        float computeAvailable = 1.0f - metrics.computeUtilization;
        float memoryAvailable = 1.0f - (static_cast<float>(metrics.memoryUsed) / metrics.memoryLimit);
        float priorityMatch = 1.0f - std::abs(desc.priority - 0) / 10.0f;  // Normalize
        float score = 0.5f * computeAvailable + 0.3f * memoryAvailable + 0.2f * priorityMatch;

        if (score > bestScore) {
            bestScore = score;
            bestContext = ctxId;
        }
    }

    return bestContext;
}

void MultiTenantManager::rebalanceWorkloads() {
    auto activeContexts = contextManager_.getActiveContexts();

    for (auto ctxId : activeContexts) {
        auto metrics = contextManager_.getContextMetrics(ctxId);

        if (metrics.computeUtilization > 0.9f) {
            contextManager_.throttleContext(ctxId, 0.8f);
        }

        if (metrics.computeUtilization < 0.3f && metrics.memoryUtilization < 0.3f) {
            contextManager_.throttleContext(ctxId, 1.0f);
        }
    }
}

std::vector<WorkloadHandle> MultiTenantManager::getActiveWorkloads() const {
    std::vector<WorkloadHandle> active;
    for (const auto& [id, handle] : handles_) {
        if (handle.status == WorkloadStatus::kPending ||
            handle.status == WorkloadStatus::kRunning) {
            active.push_back(handle);
        }
    }
    return active;
}

bool MultiTenantManager::waitForWorkload(WorkloadHandle handle, int timeoutMs) {
    if (timeoutMs < 0) {
        while (true) {
            const WorkloadStatus status = getWorkloadStatus(handle.id);
            if (status == WorkloadStatus::kCompleted) {
                return true;
            }
            if (status == WorkloadStatus::kCancelled || status == WorkloadStatus::kError) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() <= deadline) {
        const WorkloadStatus status = getWorkloadStatus(handle.id);
        if (status == WorkloadStatus::kCompleted) {
            return true;
        }
        if (status == WorkloadStatus::kCancelled || status == WorkloadStatus::kError) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

void MultiTenantManager::setRebalancingPolicy(
    bool autoRebalance,
    int rebalanceIntervalMs
) {
    autoRebalance_ = autoRebalance;
    rebalanceIntervalMs_ = std::max(1, rebalanceIntervalMs);
}

QoSManager::QoSManager(GreenContextManager& manager) : manager_(manager) {}

void QoSManager::setQoS(GreenContextId ctxId, const QoSConfig& config) {
    qosConfigs_[ctxId] = config;
}

bool QoSManager::checkQoS(GreenContextId ctxId) const {
    auto configIt = qosConfigs_.find(ctxId);
    if (configIt == qosConfigs_.end()) {
        return true;
    }

    const QoSConfig& config = configIt->second;
    auto metrics = manager_.getContextMetrics(ctxId);
    if (!metrics.active) {
        return false;
    }
    if (metrics.memoryLimit > 0) {
        const size_t minMemory = config.minMemoryMB * BYTES_PER_MB;
        const size_t maxMemory =
            config.maxMemoryMB == 0 ? metrics.memoryLimit : config.maxMemoryMB * BYTES_PER_MB;
        if (metrics.memoryLimit < minMemory || metrics.memoryUsed > maxMemory) {
            return false;
        }
    }
    return metrics.computeUtilization <= config.maxComputeFraction;
}

std::string QoSManager::getQoSReport(GreenContextId ctxId) const {
    std::ostringstream out;
    auto metrics = manager_.getContextMetrics(ctxId);
    out << "context=" << ctxId
        << " active=" << metrics.active
        << " compute=" << metrics.computeUtilization
        << " memory=" << metrics.memoryUsed << "/" << metrics.memoryLimit
        << " qos_ok=" << checkQoS(ctxId);
    return out.str();
}

void QoSManager::enforceQoS(GreenContextId ctxId) {
    auto configIt = qosConfigs_.find(ctxId);
    if (configIt == qosConfigs_.end()) {
        return;
    }

    const QoSConfig& config = configIt->second;
    auto metrics = manager_.getContextMetrics(ctxId);
    if (metrics.computeUtilization > config.targetUtilization) {
        manager_.throttleContext(ctxId, config.targetUtilization);
    } else if (metrics.computeUtilization < config.minComputeFraction) {
        manager_.throttleContext(ctxId, config.maxComputeFraction);
    }
}

void QoSManager::enableAutoQoS(int checkIntervalMs) {
    (void)checkIntervalMs;
    for (const auto& [ctxId, config] : qosConfigs_) {
        enforceQoS(ctxId);
    }
}

namespace {
GreenContextManager& globalGreenContextManager() {
    static GreenContextManager manager;
    return manager;
}
}  // namespace

GreenContext createGreenContext(
    float computeFraction,
    size_t memoryLimit,
    int priority
) {
    return globalGreenContextManager().createContext(computeFraction, memoryLimit, priority);
}

void destroyGreenContext(GreenContextId id) {
    globalGreenContextManager().destroyContext(id);
}
