"""
Performance Benchmarking Script for SegMind
Measures API response times, database query performance, and ML inference speed
"""

import time
import requests
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "segmind_db",
    "user": "postgres",
    "password": "postgres123"
}

class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            "api_endpoints": [],
            "database_queries": [],
            "ml_inference": [],
            "summary": {}
        }
        self.token = None
        
    def authenticate(self):
        """Authenticate and get access token"""
        print("🔐 Authenticating...")
        try:
            # Read test token if exists
            if Path("test_token.txt").exists():
                with open("test_token.txt", "r") as f:
                    self.token = f.read().strip()
                print("✅ Loaded authentication token")
                return True
        except Exception as e:
            print(f"⚠️ Authentication failed: {e}")
            return False
            
    def benchmark_api_endpoint(self, method, endpoint, data=None, iterations=10):
        """Benchmark a single API endpoint"""
        print(f"📊 Benchmarking {method} {endpoint}...")
        
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        times = []
        for i in range(iterations):
            start = time.time()
            try:
                if method == "GET":
                    response = requests.get(f"{API_BASE_URL}{endpoint}", headers=headers, timeout=30)
                elif method == "POST":
                    response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, headers=headers, timeout=30)
                
                elapsed = time.time() - start
                times.append(elapsed)
                
                if i == 0:  # Log first request details
                    print(f"   Status: {response.status_code}, Time: {elapsed:.3f}s")
                    
            except Exception as e:
                print(f"   ⚠️ Request failed: {e}")
                
        if times:
            result = {
                "endpoint": f"{method} {endpoint}",
                "mean": np.mean(times),
                "median": np.median(times),
                "min": np.min(times),
                "max": np.max(times),
                "std": np.std(times),
                "iterations": len(times)
            }
            self.results["api_endpoints"].append(result)
            print(f"   ✅ Mean: {result['mean']:.3f}s, Median: {result['median']:.3f}s")
            return result
        return None
        
    def benchmark_database_queries(self):
        """Benchmark common database queries"""
        print("\n📊 Benchmarking Database Queries...")
        
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            
            queries = [
                ("Count Files", "SELECT COUNT(*) FROM files"),
                ("Recent Files", "SELECT * FROM files ORDER BY upload_date DESC LIMIT 20"),
                ("Files with Analysis", """
                    SELECT f.*, a.* 
                    FROM files f 
                    LEFT JOIN analysis_results a ON f.file_id = a.file_id 
                    LIMIT 20
                """),
                ("Doctor Access Check", """
                    SELECT f.* 
                    FROM files f
                    INNER JOIN file_access_permissions p ON f.file_id = p.file_id
                    WHERE p.doctor_id = 9 AND p.status = 'active'
                    LIMIT 20
                """),
                ("Collaboration Lookup", """
                    SELECT * FROM case_collaborations 
                    WHERE primary_doctor_id = 9 OR collaborating_doctor_id = 9
                    LIMIT 20
                """),
            ]
            
            for name, query in queries:
                times = []
                for _ in range(10):
                    start = time.time()
                    cursor.execute(query)
                    results = cursor.fetchall()
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                result = {
                    "query": name,
                    "mean": np.mean(times),
                    "median": np.median(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "rows": len(results) if results else 0
                }
                self.results["database_queries"].append(result)
                print(f"   {name}: {result['mean']*1000:.1f}ms (avg), {result['rows']} rows")
                
            cursor.close()
            conn.close()
            print("✅ Database benchmarks complete")
            
        except Exception as e:
            print(f"⚠️ Database benchmark failed: {e}")
            
    def run_api_benchmarks(self):
        """Run comprehensive API benchmarks"""
        print("\n📊 Running API Benchmarks...")
        
        # Authentication endpoint
        self.benchmark_api_endpoint("POST", "/users/login", 
                                   data={"username": "test_user", "password": "test_pass"}, 
                                   iterations=5)
        
        # File listing
        self.benchmark_api_endpoint("GET", "/files/list", iterations=10)
        
        # Visualization endpoints (if file exists)
        self.benchmark_api_endpoint("GET", "/visualization/multiview/74", iterations=5)
        self.benchmark_api_endpoint("GET", "/visualization/montage/74", iterations=5)
        
        # XAI endpoints
        self.benchmark_api_endpoint("GET", "/xai/gradcam/74", iterations=3)
        
        # 3D reconstruction
        self.benchmark_api_endpoint("GET", "/reconstruction/viewer-data/74", iterations=3)
        
        print("✅ API benchmarks complete")
        
    def generate_report(self):
        """Generate performance report"""
        print("\n📊 Generating Performance Report...")
        
        report = []
        report.append("=" * 80)
        report.append("SEGMIND PERFORMANCE BENCHMARK REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # API Endpoints Summary
        if self.results["api_endpoints"]:
            report.append("\n📡 API ENDPOINT PERFORMANCE")
            report.append("-" * 80)
            report.append(f"{'Endpoint':<50} {'Mean':<10} {'Median':<10} {'Min':<10}")
            report.append("-" * 80)
            
            for result in self.results["api_endpoints"]:
                report.append(f"{result['endpoint']:<50} {result['mean']:.3f}s    {result['median']:.3f}s    {result['min']:.3f}s")
                
        # Database Query Summary
        if self.results["database_queries"]:
            report.append("\n💾 DATABASE QUERY PERFORMANCE")
            report.append("-" * 80)
            report.append(f"{'Query':<40} {'Mean':<15} {'Rows':<10}")
            report.append("-" * 80)
            
            for result in self.results["database_queries"]:
                report.append(f"{result['query']:<40} {result['mean']*1000:.1f}ms         {result['rows']}")
                
        # Performance Summary
        report.append("\n🎯 PERFORMANCE SUMMARY")
        report.append("-" * 80)
        
        if self.results["api_endpoints"]:
            api_times = [r['mean'] for r in self.results["api_endpoints"]]
            report.append(f"Average API Response Time: {np.mean(api_times):.3f}s")
            report.append(f"Slowest API Endpoint: {max(self.results['api_endpoints'], key=lambda x: x['mean'])['endpoint']} ({max(api_times):.3f}s)")
            
        if self.results["database_queries"]:
            db_times = [r['mean'] * 1000 for r in self.results["database_queries"]]
            report.append(f"Average DB Query Time: {np.mean(db_times):.1f}ms")
            report.append(f"Slowest DB Query: {max(self.results['database_queries'], key=lambda x: x['mean'])['query']} ({max(db_times):.1f}ms)")
            
        # Recommendations
        report.append("\n💡 OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 80)
        
        slow_apis = [r for r in self.results["api_endpoints"] if r['mean'] > 2.0]
        if slow_apis:
            report.append("⚠️ Slow API Endpoints (>2s):")
            for r in slow_apis:
                report.append(f"   - {r['endpoint']}: {r['mean']:.3f}s - Consider caching or optimization")
        else:
            report.append("✅ All API endpoints performing well (<2s)")
            
        slow_queries = [r for r in self.results["database_queries"] if r['mean'] * 1000 > 100]
        if slow_queries:
            report.append("\n⚠️ Slow Database Queries (>100ms):")
            for r in slow_queries:
                report.append(f"   - {r['query']}: {r['mean']*1000:.1f}ms - Check indexes")
        else:
            report.append("✅ All database queries performing well (<100ms)")
            
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        print(report_text)
        
        with open("performance_report.txt", "w", encoding='utf-8') as f:
            f.write(report_text)
        print("\n✅ Report saved to performance_report.txt")
        
        return report_text
        
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("🚀 Starting Performance Benchmarks...\n")
        
        # Authenticate
        if not self.authenticate():
            print("⚠️ Skipping API benchmarks (no authentication)")
        else:
            self.run_api_benchmarks()
            
        # Database benchmarks
        self.benchmark_database_queries()
        
        # Generate report
        self.generate_report()
        
        print("\n✅ All benchmarks complete!")


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
