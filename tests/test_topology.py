"""Tests for topology module."""

import unittest
from llm_perf.hardware.topology import (
    NetworkTopology,
    TopologyLevel,
    TopologyType,
    ClosTopologyBuilder,
)
from llm_perf.hardware import Cluster
from llm_perf.hardware.device import Device
from llm_perf.strategy.base import StrategyConfig, SPType


class TestTopologyLevel(unittest.TestCase):
    """Test TopologyLevel class."""
    
    def test_basic_creation(self):
        """Test creating a topology level."""
        level = TopologyLevel(
            name="node",
            level=0,
            bandwidth_gbps=900.0,
            latency_us=1.0,
            devices_per_group=8,
        )
        
        self.assertEqual(level.name, "node")
        self.assertEqual(level.level, 0)
        self.assertEqual(level.bandwidth_gbps, 900.0)
        self.assertEqual(level.latency_us, 1.0)
        self.assertEqual(level.devices_per_group, 8)
    
    def test_invalid_level(self):
        """Test that negative level raises error."""
        with self.assertRaises(ValueError):
            TopologyLevel(
                name="invalid",
                level=-1,
                bandwidth_gbps=100.0,
            )


class TestNetworkTopology(unittest.TestCase):
    """Test NetworkTopology class."""
    
    def test_create_2tier_simple(self):
        """Test creating simple 2-tier topology."""
        topo = NetworkTopology.create_2tier_simple(
            intra_node_bw_gbps=900.0,
            inter_node_bw_gbps=200.0,
            devices_per_node=8,
        )
        
        self.assertEqual(topo.topology_type, TopologyType.SWITCH)
        self.assertEqual(len(topo.levels), 2)
        
        # Check level 0 (node)
        self.assertEqual(topo.levels[0].level, 0)
        self.assertEqual(topo.levels[0].bandwidth_gbps, 900.0)
        self.assertEqual(topo.levels[0].devices_per_group, 8)
        
        # Check level 1 (inter-node)
        self.assertEqual(topo.levels[1].level, 1)
        self.assertEqual(topo.levels[1].bandwidth_gbps, 200.0)
    
    def test_create_clos_3tier(self):
        """Test creating 3-tier Clos topology."""
        topo = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
            devices_per_node=8,
            nodes_per_rack=16,
            racks_per_cluster=32,
        )
        
        self.assertEqual(topo.topology_type, TopologyType.CLOS)
        self.assertEqual(len(topo.levels), 3)
        
        # Check bandwidth decreases with level
        self.assertEqual(topo.levels[0].bandwidth_gbps, 900.0)
        self.assertEqual(topo.levels[1].bandwidth_gbps, 200.0)
        self.assertEqual(topo.levels[2].bandwidth_gbps, 100.0)
        
        # Check devices per group increases
        self.assertEqual(topo.levels[0].devices_per_group, 8)
        self.assertEqual(topo.levels[1].devices_per_group, 128)  # 8*16
        self.assertEqual(topo.levels[2].devices_per_group, 4096)  # 8*16*32
    
    def test_create_fat_tree(self):
        """Test creating Fat-Tree topology."""
        topo = NetworkTopology.create_fat_tree(
            core_bw_gbps=100.0,
            agg_bw_gbps=400.0,
            edge_bw_gbps=800.0,
            oversubscription=4.0,
        )
        
        self.assertEqual(topo.topology_type, TopologyType.CLOS)
        self.assertEqual(topo.levels[0].name, "edge")
        self.assertEqual(topo.levels[1].name, "aggregation")
        self.assertEqual(topo.levels[2].name, "core")
    
    def test_get_level_for_distance(self):
        """Test finding topology level for communication distance."""
        topo = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
            devices_per_node=8,
            nodes_per_rack=16,
            racks_per_cluster=32,
        )
        
        # Distance 0-7: node level
        level = topo.get_level_for_distance(4)
        self.assertEqual(level.level, 0)
        
        # Distance 8-127: rack level
        level = topo.get_level_for_distance(64)
        self.assertEqual(level.level, 1)
        
        # Distance 128+: cluster level
        level = topo.get_level_for_distance(200)
        self.assertEqual(level.level, 2)
    
    def test_legacy_backward_compatibility(self):
        """Test backward compatibility with old NetworkConfig."""
        topo = NetworkTopology(
            name="legacy",
            intra_node_bandwidth_gbps=900.0,
            inter_node_bandwidth_gbps=200.0,
        )
        
        self.assertEqual(len(topo.levels), 2)
        self.assertEqual(topo.levels[0].bandwidth_gbps, 900.0)
        self.assertEqual(topo.levels[1].bandwidth_gbps, 200.0)


class TestClosTopologyBuilder(unittest.TestCase):
    """Test ClosTopologyBuilder class."""
    
    def test_calculate_clos_parameters_small(self):
        """Test Clos parameters for small cluster."""
        params = ClosTopologyBuilder.calculate_clos_parameters(
            num_devices=32,
            switch_radix=32,
        )
        
        # 32 devices fit in single tier
        self.assertEqual(params["num_tiers"], 1)
    
    def test_calculate_clos_parameters_medium(self):
        """Test Clos parameters for medium cluster."""
        params = ClosTopologyBuilder.calculate_clos_parameters(
            num_devices=256,
            switch_radix=32,
        )
        
        # 256 devices need 2-tier (leaf-spine)
        self.assertEqual(params["num_tiers"], 2)
    
    def test_calculate_clos_parameters_large(self):
        """Test Clos parameters for large cluster."""
        params = ClosTopologyBuilder.calculate_clos_parameters(
            num_devices=1024,
            switch_radix=32,
        )
        
        # 1024 devices need 3-tier
        self.assertEqual(params["num_tiers"], 3)
    
    def test_create_from_device_count_2tier(self):
        """Test automatic 2-tier Clos creation."""
        topo = ClosTopologyBuilder.create_from_device_count(
            num_devices=128,
            host_bw_gbps=200.0,
            leaf_bw_gbps=200.0,
            spine_bw_gbps=200.0,
        )
        
        self.assertEqual(topo.topology_type, TopologyType.CLOS)
        self.assertEqual(len(topo.levels), 2)
    
    def test_create_from_device_count_3tier(self):
        """Test automatic 3-tier Clos creation."""
        topo = ClosTopologyBuilder.create_from_device_count(
            num_devices=1024,
            switch_radix=32,
            host_bw_gbps=200.0,
            leaf_bw_gbps=200.0,
            spine_bw_gbps=200.0,
        )
        
        self.assertEqual(topo.topology_type, TopologyType.CLOS)
        self.assertEqual(len(topo.levels), 3)


class TestClusterWithTopology(unittest.TestCase):
    """Test Cluster class with hierarchical topology."""
    
    def test_cluster_with_clos_topology(self):
        """Test creating cluster with Clos topology."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )
        
        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=256,
            topology=topology,
            devices_per_node=8,
        )
        
        self.assertEqual(cluster.num_devices, 256)
        self.assertEqual(len(cluster.topology.levels), 3)
    
    def test_bandwidth_between_same_node(self):
        """Test bandwidth for same-node communication."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
            devices_per_node=8,
        )
        
        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=64,
            topology=topology,
            devices_per_node=8,
        )
        
        # Ranks 0 and 1 are on same node
        bw = cluster.get_bandwidth_between(0, 1)
        self.assertEqual(bw, 900.0)
    
    def test_bandwidth_between_different_nodes(self):
        """Test bandwidth for cross-node communication."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
            devices_per_node=8,
        )
        
        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=64,
            topology=topology,
            devices_per_node=8,
        )
        
        # Ranks 0 and 16 are on different nodes (different rack)
        bw = cluster.get_bandwidth_between(0, 64)
        # Should get rack-level bandwidth
        self.assertLess(bw, 900.0)
    
    def test_estimate_allreduce_with_topology(self):
        """Test AllReduce time estimation with topology."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )
        
        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=256,
            topology=topology,
            devices_per_node=8,
        )
        
        # Estimate AllReduce for 1GB data
        time = cluster.estimate_allreduce_time(
            num_bytes=1_000_000_000,
            participating_ranks=list(range(256))
        )
        
        # Should be positive and reasonable
        self.assertGreater(time, 0)
        # With hierarchical topology, should be slower than flat 900 GB/s
        # but faster than flat 100 GB/s
    
    def test_create_with_clos_topology_helper(self):
        """Test the helper method for creating Clos cluster."""
        cluster = Cluster.create_with_clos_topology(
            preset_name="A100-SXM-80GB",
            num_devices=128,
            host_bw_gbps=200.0,
            leaf_bw_gbps=200.0,
            spine_bw_gbps=200.0,
            switch_radix=32,
            oversubscription=4.0,
        )

        self.assertEqual(cluster.num_devices, 128)
        self.assertEqual(cluster.topology.topology_type, TopologyType.CLOS)

    def test_build_communication_domain_groups(self):
        """Test building communication domain groups."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )

        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=16,
            topology=topology,
            devices_per_node=8,
        )

        # Create strategy config
        strategy = StrategyConfig(tp_degree=4, pp_degree=2, dp_degree=2)

        # Build communication domain groups
        groups = cluster.build_communication_domain_groups(strategy)

        # Check TP groups
        self.assertIn("tp_groups", groups)
        self.assertEqual(len(groups["tp_groups"]), 4)  # 2 PP * 2 DP
        self.assertEqual(len(groups["tp_groups"][0]), 4)  # Each TP group has 4 ranks

        # Check PP groups
        self.assertIn("pp_groups", groups)
        self.assertEqual(len(groups["pp_groups"]), 8)  # 4 TP * 2 DP positions

        # Check DP groups
        self.assertIn("dp_groups", groups)
        self.assertEqual(len(groups["dp_groups"]), 8)  # 4 TP * 2 PP positions

    def test_get_bandwidth_for_communication_domain(self):
        """Test getting bandwidth for communication domain types."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )

        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=64,
            topology=topology,
            devices_per_node=8,
        )

        # TP should use level 0 (node) bandwidth
        tp_bw = cluster.get_bandwidth_for_communication_domain("tp")
        self.assertEqual(tp_bw, 900.0)

        # DP should use level 1 (rack) bandwidth by default
        dp_bw = cluster.get_bandwidth_for_communication_domain("dp")
        self.assertEqual(dp_bw, 200.0)

        # PP should use level 1 (rack) bandwidth by default
        pp_bw = cluster.get_bandwidth_for_communication_domain("pp")
        self.assertEqual(pp_bw, 200.0)

    def test_get_bandwidth_for_communication_domain_with_strategy(self):
        """Test bandwidth lookup with strategy-based level adjustment."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )

        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=256,
            topology=topology,
            devices_per_node=8,
        )

        # Small TP (fits in node)
        strategy_small_tp = StrategyConfig(tp_degree=4)
        bw = cluster.get_bandwidth_for_communication_domain("tp", strategy_small_tp)
        self.assertEqual(bw, 900.0)  # Level 0 (node)

        # Large DP (crosses rack)
        strategy_large_dp = StrategyConfig(dp_degree=32)
        bw = cluster.get_bandwidth_for_communication_domain("dp", strategy_large_dp)
        self.assertEqual(bw, 100.0)  # Level 2 (cluster/inter_rack)

    def test_get_bandwidth_for_topology_level(self):
        """Test getting bandwidth by topology level name."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )

        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=64,
            topology=topology,
            devices_per_node=8,
        )

        # Test various topology level names
        self.assertEqual(cluster.get_bandwidth_for_topology_level("node"), 900.0)
        self.assertEqual(cluster.get_bandwidth_for_topology_level("intra_node"), 900.0)
        self.assertEqual(cluster.get_bandwidth_for_topology_level("rack"), 200.0)
        self.assertEqual(cluster.get_bandwidth_for_topology_level("intra_rack"), 200.0)
        self.assertEqual(cluster.get_bandwidth_for_topology_level("cluster"), 100.0)
        self.assertEqual(cluster.get_bandwidth_for_topology_level("inter_rack"), 100.0)

    def test_communication_domain_priority(self):
        """Test communication domain bandwidth priority ordering."""
        topology = NetworkTopology.create_clos_3tier(
            node_bw_gbps=900.0,
            rack_bw_gbps=200.0,
            cluster_bw_gbps=100.0,
        )

        cluster = Cluster.create_from_preset(
            preset_name="H100-SXM-80GB",
            num_devices=256,
            topology=topology,
            devices_per_node=8,
        )

        strategy = StrategyConfig(tp_degree=4, ep_degree=4, dp_degree=4, pp_degree=4)
        mapping = strategy.get_communication_domain_mapping(devices_per_node=8)

        # Priority: TP > EP > Ulysses > Ring > DP > PP
        # Higher bandwidth means higher priority (lower level)
        tp_level = mapping["tp"]["topology_level"]
        ep_level = mapping["ep"]["topology_level"]
        dp_level = mapping["dp"]["topology_level"]
        pp_level = mapping["pp"]["topology_level"]

        # TP should have lowest (best) topology level
        self.assertLessEqual(tp_level, ep_level)
        self.assertLessEqual(tp_level, dp_level)
        self.assertLessEqual(tp_level, pp_level)


if __name__ == "__main__":
    unittest.main()
