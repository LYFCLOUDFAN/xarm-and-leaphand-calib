from pathlib import Path
import trimesh

def get_cvx_hull(mesh_path):
    mesh = trimesh.load(mesh_path)
    return mesh.convex_hull

# Example usage:
meshes_dir_path = Path(__file__).parent / "meshes" 
# 遍历meshes目录中的所有文件
for mesh_file in meshes_dir_path.glob('*'):
    # 检查文件是否为STL或其他3D模型格式
    if mesh_file.suffix.lower() in ['.stl', '.obj', '.ply']:
        # 获取凸包
        cvx_hull = get_cvx_hull(str(mesh_file))
        
        # 创建新的文件名，添加"_cvx_hull"后缀
        new_filename = mesh_file.stem + '_cvx_hull' + mesh_file.suffix
        new_filepath = meshes_dir_path / new_filename
        
        # 保存凸包
        cvx_hull.export(str(new_filepath))
        
        print(f"已为 {mesh_file.name} 创建凸包并保存为 {new_filename}")
