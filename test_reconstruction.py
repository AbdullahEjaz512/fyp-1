"""
Test Module 8: 3D Tumor Reconstruction
Tests mesh generation, export, and visualization data
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TOKEN_FILE = "test_token.txt"

def get_token():
    """Get authentication token"""
    if Path(TOKEN_FILE).exists():
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()
    
    # Login
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={
            "email": "test_doctor@example.com",
            "password": "password123"
        }
    )
    
    if response.status_code == 200:
        token = response.json()['access_token']
        with open(TOKEN_FILE, 'w') as f:
            f.write(token)
        return token
    else:
        raise Exception("Authentication failed")

def test_reconstruction():
    """Test 3D reconstruction endpoints"""
    
    print("=" * 60)
    print("MODULE 8: 3D TUMOR RECONSTRUCTION TEST")
    print("=" * 60)
    
    token = get_token()
    headers = {'Authorization': f'Bearer {token}'}
    
    # Assuming file_id=1 exists (adjust as needed)
    file_id = 1
    
    print(f"\nTesting with file_id={file_id}")
    print("-" * 60)
    
    # Test 1: Generate 3D mesh
    print("\n1. Testing mesh generation...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/reconstruction/generate/{file_id}",
            headers=headers,
            params={'smoothing': True, 'step_size': 2}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Mesh generated successfully")
            print(f"   - Regions: {data['num_regions']}")
            print(f"   - Surface area: {data['total_surface_area_mm2']:.2f} mm²")
            
            if 'meshes' in data:
                for region_name, mesh in data['meshes'].items():
                    print(f"\n   {region_name}:")
                    print(f"     - Vertices: {mesh['num_vertices']:,}")
                    print(f"     - Faces: {mesh['num_faces']:,}")
                    print(f"     - Surface: {mesh['surface_area_mm2']:.2f} mm²")
        else:
            print(f"   ✗ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 2: Get mesh data for specific region
    print("\n2. Testing mesh data retrieval...")
    try:
        for region in ['NCR', 'ED', 'ET']:
            response = requests.get(
                f"{BASE_URL}/api/v1/reconstruction/mesh/{file_id}/{region}",
                headers=headers,
                params={'format': 'json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✓ {region} mesh data retrieved")
                print(f"     - Vertices: {data['num_vertices']:,}")
                print(f"     - Faces: {data['num_faces']:,}")
            else:
                print(f"   ✗ {region} failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 3: Get viewer data
    print("\n3. Testing viewer data (VTK.js)...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/reconstruction/viewer-data/{file_id}",
            headers=headers,
            params={'format': 'vtkjs'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ VTK.js viewer data generated")
            print(f"   - Regions: {len(data.get('regions', []))}")
            print(f"   - Format: {data.get('vtkClass', 'N/A')}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 4: Get viewer data (Three.js)
    print("\n4. Testing viewer data (Three.js)...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/reconstruction/viewer-data/{file_id}",
            headers=headers,
            params={'format': 'threejs'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Three.js viewer data generated")
            print(f"   - Geometries: {len(data.get('geometries', []))}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 5: Export STL
    print("\n5. Testing STL export...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/reconstruction/export/stl/{file_id}/ET",
            headers=headers
        )
        
        if response.status_code == 200:
            stl_size = len(response.content)
            print(f"   ✓ STL export successful")
            print(f"   - Size: {stl_size / 1024:.2f} KB")
            
            # Save sample
            output_path = Path("data/3d_exports/test_ET.stl")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   - Saved to: {output_path}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 6: Export OBJ
    print("\n6. Testing OBJ export...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/reconstruction/export/obj/{file_id}/ET",
            headers=headers
        )
        
        if response.status_code == 200:
            obj_size = len(response.content)
            print(f"   ✓ OBJ export successful")
            print(f"   - Size: {obj_size / 1024:.2f} KB")
            
            # Save sample
            output_path = Path("data/3d_exports/test_ET.obj")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   - Saved to: {output_path}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 7: Get 3D statistics
    print("\n7. Testing 3D statistics...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/reconstruction/stats/{file_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Statistics retrieved")
            print(f"   - Total vertices: {data['total_vertices']:,}")
            print(f"   - Total faces: {data['total_faces']:,}")
            print(f"   - Surface area: {data['total_surface_area_mm2']:.2f} mm²")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 8: Get cross-sections
    print("\n8. Testing cross-sections...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/reconstruction/cross-sections/{file_id}",
            headers=headers,
            params={'num_slices': 10}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Cross-sections generated")
            cross_sections = data.get('cross_sections', {})
            print(f"   - Number of slices: {cross_sections.get('num_slices', 0)}")
            print(f"   - Z range: {cross_sections.get('z_range', [0, 0])}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("MODULE 8 TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start backend: python backend/app/main.py")
    print("2. Start frontend: cd frontend && npm run dev")
    print("3. Navigate to: http://localhost:5173/reconstruction?file_id=1")
    print("=" * 60)


if __name__ == "__main__":
    test_reconstruction()
