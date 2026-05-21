"""Tests for the agent-friendly CLI enhancements."""
import sys
import os
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cli import parse_nlp_query

class TestCLI(unittest.TestCase):
    def test_parse_nlp_query_model_sizes(self):
        """Test model size extraction from NLP queries."""
        cases = {
            "Evaluate Llama-3-70B on 1xH100": 70.0,
            "deepseek 671b benchmark": 671.0,
            "evaluate 8 billion model": 8.0,
            "1.5 b params": 1.5,
            "qwen-2.5-7b-instruct": 7.0
        }
        for query, expected in cases.items():
            parsed = parse_nlp_query(query)
            self.assertEqual(parsed.get("model"), expected, f"Failed for query: {query}")

    def test_parse_nlp_query_hardware_and_engines(self):
        """Test hardware and engine extraction from NLP queries."""
        cases = [
            ("Evaluate on 1xA100 using vllm", {"hw": "1xA100", "engine": "vllm"}),
            ("run on rtx-3090 with pytorch", {"hw": "RTX-3090", "engine": "pytorch"}),
            ("benchmarking 1xH200 with tensorrt", {"hw": "1xH200", "engine": "tensorrt"}),
            ("openvino on 32vcpu-c7i", {"hw": "32vCPU-C7i", "engine": "openvino"})
        ]
        for query, expected in cases:
            parsed = parse_nlp_query(query)
            for k, v in expected.items():
                self.assertEqual(parsed.get(k), v, f"Failed matching key '{k}' for query: {query}")

    def test_parse_nlp_query_parameters(self):
        """Test batch sizes, precision, inputs, and outputs extraction."""
        query = "Evaluate 70b on 1xH100 with batch size 32, prompt length 512, generate 128, 4 bits"
        parsed = parse_nlp_query(query)
        self.assertEqual(parsed.get("model"), 70.0)
        self.assertEqual(parsed.get("hw"), "1xH100")
        self.assertEqual(parsed.get("batch"), 32)
        self.assertEqual(parsed.get("p_in"), 512)
        self.assertEqual(parsed.get("p_out"), 128)
        self.assertEqual(parsed.get("bits"), 4)

    def test_parse_nlp_query_recommend_intent(self):
        """Test sweep recommendation intent detection from keywords."""
        recommend_queries = [
            "recommend a setup for 8b",
            "sweep configurations for llama 70b",
            "find the best deployment for deepseek",
            "what is the cheapest setup for 7b",
            "fastest GPU configuration"
        ]
        for query in recommend_queries:
            parsed = parse_nlp_query(query)
            self.assertTrue(parsed.get("recommend", False), f"Failed to detect recommend intent for: {query}")

if __name__ == "__main__":
    unittest.main()
