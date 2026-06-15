from __future__ import annotations

import numpy as np
import pytest
from pynerve.exceptions import ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_sparse_ph_layers_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.nn.sparse_ph import (
        SparsePH,
        TopologyAttention,
        WindowedPH,
        compute_witness_persistence,
        farthest_point_sampling,
    )

    landmarks, indices = farthest_point_sampling(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 2
    )
    assert landmarks.shape == (2, 2), f"expected (2, 2), got {landmarks.shape}"
    assert indices.shape == (2,), f"expected (2,), got {indices.shape}"
    with pytest.raises(ValueError, match="n_samples"):
        farthest_point_sampling(torch.ones(2, 2), -1)
    with pytest.raises(ValidationError, match="points"):
        farthest_point_sampling(torch.tensor([[float("nan"), 0.0]]), 1)

    with pytest.raises(ValueError, match="max_radius"):
        SparsePH(max_radius=float("nan"))
    with pytest.raises(ValueError, match="landmark_ratio"):
        SparsePH(landmark_ratio=float("inf"))
    with pytest.raises(ValidationError, match="points"):
        SparsePH()(torch.tensor([[[float("nan"), 0.0]]]))

    with pytest.raises(ValueError, match="landmarks"):
        compute_witness_persistence(
            np.asarray([[float("nan"), 0.0]]),
            np.ones((1, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="max_dim"):
        compute_witness_persistence(
            np.ones((1, 2), dtype=np.float64),
            np.ones((1, 2), dtype=np.float64),
            max_dim=float("nan"),
        )
    with pytest.raises(ValueError, match="same dimension"):
        compute_witness_persistence(
            np.ones((1, 2), dtype=np.float64),
            np.ones((1, 3), dtype=np.float64),
        )

    windowed = WindowedPH(window_size=4, stride=1, max_dim=0)
    assert windowed(torch.ones(2, 3, 2)).shape == (2, 0), (
        f"expected (2, 0), got {windowed(torch.ones(2, 3, 2)).shape}"
    )
    assert (
        torch.isfinite(
            windowed._diagrams_to_features([torch.tensor([[0.0, float("inf")], [0.0, 0.5]])])
        )
        .all()
        .item()
    ), "expected all finite diagram features"
    with pytest.raises(ValidationError, match="window_size"):
        WindowedPH(window_size=0)
    with pytest.raises(ValidationError, match="points"):
        windowed(torch.tensor([[[float("nan"), 0.0]]]))
    with pytest.raises(ValueError, match="deaths"):
        windowed._diagrams_to_features([torch.tensor([[1.0, 0.0]])])

    attention = TopologyAttention(n_heads=2, dim=4, n_clusters=2)
    assert torch.isfinite(attention(torch.ones(1, 3, 4))).all().item(), (
        "expected all finite attention output"
    )
    with pytest.raises(ValueError, match="dropout"):
        TopologyAttention(n_heads=2, dim=4, dropout=float("nan"))
    with pytest.raises(ValidationError, match="x"):
        attention(torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]))
    with pytest.raises(ValidationError, match="mask"):
        attention(torch.ones(1, 1, 4), mask=torch.tensor([[[[float("nan")]]]]))
    with pytest.raises(ValueError, match="attention target"):
        attention(torch.ones(1, 1, 4), mask=torch.zeros(1, 1, 1, 1))
    with torch.no_grad():
        attention.cluster_centers[0, 0] = float("nan")
    with pytest.raises(ValidationError, match="cluster_centers"):
        attention(torch.ones(1, 3, 4))


def test_torch_mapper_rejects_invalid_numeric_inputs(torch) -> None:
    from pynerve.torch.mapper import MapperTransformer, mapper

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    result = mapper(
        points,
        filter_function=lambda x: x[:, :1],
        cover_resolution=2,
        clusterer="connected",
        dbscan_min_samples=1,
        return_graph=False,
    )
    assert set(result) >= {"nodes", "edges", "filter_values"}, (
        f"expected keys {{'nodes', 'edges', 'filter_values'}}, got {set(result)}"
    )
    assert torch.isfinite(result["filter_values"]).all().item(), "expected all finite filter values"

    with pytest.raises(Exception, match="point_cloud"):
        mapper(torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(Exception, match="dbscan_eps"):
        mapper(points, dbscan_eps=float("nan"))
    with pytest.raises(Exception, match="cover_resolution"):
        mapper(points, cover_resolution=2.5)
    with pytest.raises(Exception, match="filter_function"):
        mapper(points, filter_function="not_a_filter")
    with pytest.raises(Exception, match="filter values"):
        mapper(
            points,
            filter_function=lambda _x: torch.tensor([0.0, float("nan"), 1.0]),
        )

    transformer = MapperTransformer(filter_function="identity")
    transformer.mapper_result_ = {
        "nodes": [{"id": 0, "filter_centroid": torch.zeros(2)}],
    }
    assert transformer.transform(points).shape == (3,), (
        f"expected (3,), got {transformer.transform(points).shape}"
    )
    with pytest.raises(Exception, match="point_cloud"):
        transformer.transform(torch.tensor([[float("nan"), 0.0]]))


def test_torch_nn_layers_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.torch.nn_layers import (
        DiagramPooling,
        PersistenceLayer,
        TopologicalAttention,
        make_topo_network,
    )

    pooling = DiagramPooling(method="mean")
    assert torch.isfinite(pooling(torch.ones(2, 3, 2))).all().item(), (
        "expected all finite diagram pooling output"
    )
    with pytest.raises(ValidationError, match="diagrams"):
        pooling(torch.tensor([float("nan")]))

    with pytest.raises(ValidationError, match="x"):
        PersistenceLayer()(torch.tensor([[float("nan"), 0.0]]))

    attention = TopologicalAttention(feature_dim=4, n_heads=2, dropout=0.0)
    output, weights = attention(torch.ones(1, 3, 4))
    assert torch.isfinite(output).all().item(), "expected all finite attention output"
    assert torch.isfinite(weights).all().item(), "expected all finite attention weights"
    with pytest.raises(ValueError, match="dropout"):
        TopologicalAttention(feature_dim=4, n_heads=2, dropout=float("nan"))

    with pytest.raises(ValidationError, match="input_dim"):
        make_topo_network(input_dim=float("nan"))
    with pytest.raises(ValueError, match="dropout"):
        make_topo_network(input_dim=2, dropout=float("inf"))


def test_nn_diagram_layers_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.nn.diagram_conv import (
        DiagramConv1D,
        DiagramConvNet,
        DiagramDeepSet,
        DiagramMultiHeadAttention,
        DiagramPooling,
    )

    diagram = torch.tensor([[[0.0, 1.0], [0.2, 0.5]]], dtype=torch.float32)

    attention = DiagramMultiHeadAttention(d_model=4, num_heads=2, dropout=0.0)
    assert torch.isfinite(attention(diagram, torch.ones(1, 2, 4))).all().item(), (
        "expected all finite multi-head attention output"
    )
    with pytest.raises(ValueError, match="dropout"):
        DiagramMultiHeadAttention(d_model=4, num_heads=2, dropout=float("nan"))
    with pytest.raises(ValidationError, match="diagram"):
        attention(torch.tensor([[[float("nan"), 1.0]]]), torch.ones(1, 1, 4))
    with pytest.raises(ValidationError, match="features"):
        attention(diagram, torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]))
    with pytest.raises(ValueError, match="deaths"):
        attention(torch.tensor([[[1.0, 0.0]]]), torch.ones(1, 1, 4))
    with pytest.raises(ValueError, match="attention target"):
        attention(diagram, torch.ones(1, 2, 4), mask=torch.zeros(1, 2))

    conv = DiagramConv1D(in_channels=0, out_channels=2, kernel_size=3)
    assert torch.isfinite(conv(diagram)).all().item(), "expected all finite conv1d output"
    with pytest.raises(ValueError, match="in_channels"):
        DiagramConv1D(in_channels=float("nan"), out_channels=2)
    with pytest.raises(ValidationError, match="diagram"):
        conv(torch.tensor([[[float("nan"), 1.0]]]))

    deepset = DiagramDeepSet(in_channels=0, hidden_channels=[4], out_channels=2)
    assert torch.isfinite(deepset(diagram)).all().item(), "expected all finite deepset output"
    with pytest.raises(ValidationError, match="hidden_channels"):
        DiagramDeepSet(in_channels=0, hidden_channels=[float("nan")], out_channels=2)

    pooling = DiagramPooling(in_channels=2, out_channels=3, num_prototypes=2)
    assert torch.isfinite(pooling(diagram, torch.ones(1, 2, 2))).all().item(), (
        "expected all finite diagram pooling output"
    )
    with pytest.raises(ValidationError, match="features"):
        pooling(diagram, torch.tensor([[[float("nan"), 0.0]]]))

    net = DiagramConvNet(in_channels=2, hidden_channels=[4], out_dim=2, num_prototypes=2)
    assert torch.isfinite(net(diagram)).all().item(), "expected all finite conv net output"
    with pytest.raises(ValidationError, match="hidden_channels"):
        DiagramConvNet(in_channels=2, hidden_channels=[float("nan")])


def test_learnable_mapper_modules_reject_invalid_numeric_inputs(torch) -> None:
    from pynerve.mapper.learnable_mapper import (
        AdaptiveCover,
        DifferentiableMapper,
        LensFunction,
        MapperAutoencoder,
        MapperGraphEncoder,
        SoftClusterAssignment,
    )
    from pynerve.mapper.mapper_gnn import MapperGNNClassifier, MapperGraphConv

    lens = LensFunction(2, output_dim=1, hidden_dims=[4])
    assert lens(torch.ones(3, 2)).shape == (3, 1), (
        f"expected (3, 1), got {lens(torch.ones(3, 2)).shape}"
    )
    with pytest.raises(ValidationError, match="input_dim"):
        LensFunction(1.5)
    with pytest.raises(ValueError, match="x"):
        lens(torch.tensor([[float("nan"), 0.0]], dtype=torch.float32))

    cover = AdaptiveCover(min_resolution=1, max_resolution=2)
    assert len(cover.create_cover(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))) >= 1, (
        f"expected at least 1, got {len(cover.create_cover(torch.tensor([[0.0, 0.0], [1.0, 1.0]])))}"
    )
    with pytest.raises(ValueError, match="min_overlap"):
        AdaptiveCover(min_overlap=float("nan"))
    with pytest.raises(TypeError, match="lens_values"):
        cover.create_cover(torch.tensor([[0, 1]], dtype=torch.int64))

    assignment = SoftClusterAssignment(temperature=0.5)
    assert assignment(torch.ones(2, 2), torch.zeros(1, 2)).shape == (2, 1), (
        f"expected (2, 1), got {assignment(torch.ones(2, 2), torch.zeros(1, 2)).shape}"
    )
    with pytest.raises(ValueError, match="temperature"):
        SoftClusterAssignment(temperature=float("nan"))

    graph_encoder = MapperGraphEncoder(2, hidden_dim=4, output_dim=3, num_layers=1)
    assert graph_encoder(torch.ones(2, 2), torch.zeros((2, 0), dtype=torch.long)).shape == (3,), (
        f"expected (3,), got {graph_encoder(torch.ones(2, 2), torch.zeros((2, 0), dtype=torch.long)).shape}"
    )
    with pytest.raises(TypeError, match="edges"):
        graph_encoder(torch.ones(2, 2), torch.tensor([[0.0], [1.0]]))

    mapper = DifferentiableMapper(
        2, lens_output_dim=1, min_cover_resolution=1, max_cover_resolution=1
    )
    assert mapper(torch.ones(1, 3, 2))["graph_embedding"].shape == (1, 32), (
        f"expected (1, 32), got {mapper(torch.ones(1, 3, 2))['graph_embedding'].shape}"
    )
    with pytest.raises(ValueError, match="data"):
        mapper(torch.tensor([[[float("nan"), 0.0]]], dtype=torch.float32))

    autoencoder = MapperAutoencoder(2, latent_dim=3, decoder_hidden_dims=[4])
    with pytest.raises(ValueError, match="latent"):
        autoencoder.decode(torch.ones(3))

    conv = MapperGraphConv(2, 3)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    assert conv(torch.ones(2, 2), edges).shape == (2, 3), (
        f"expected (2, 3), got {conv(torch.ones(2, 2), edges).shape}"
    )
    with pytest.raises(ValueError, match="edge_weights"):
        conv(torch.ones(2, 2), edges, torch.tensor([float("nan")]))

    classifier = MapperGNNClassifier(
        2, hidden_dim=4, num_classes=2, num_gnn_layers=1, use_hierarchical=False
    )
    assert classifier(torch.ones(2, 2), torch.zeros((2, 0), dtype=torch.long)).shape == (2,), (
        f"expected (2,), got {classifier(torch.ones(2, 2), torch.zeros((2, 0), dtype=torch.long)).shape}"
    )
    with pytest.raises(TypeError, match="batch_vector"):
        classifier(
            torch.ones(2, 2),
            torch.zeros((2, 0), dtype=torch.long),
            batch_vector=torch.tensor([0.0, 1.0]),
        )
