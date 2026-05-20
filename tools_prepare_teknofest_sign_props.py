from pathlib import Path
import json
import shutil
import subprocess
import sys


PROJECT_ROOT = Path.cwd()
TEXTURE_DIR = PROJECT_ROOT / "autonomous_driving" / "assets" / "traffic_signs" / "textures_white_1024"

OUT_ROOT = PROJECT_ROOT / "carla_import_packages"
PACKAGE_NAME = "TeknofestSigns"
PACKAGE_ROOT = OUT_ROOT / PACKAGE_NAME
PROPS_ROOT = PACKAGE_ROOT / "Props"
JSON_PATH = PACKAGE_ROOT / f"{PACKAGE_NAME}.json"

SIGNS = [
    "dur",
    "yol_ver",
    "girisi_olmayan_yol",
    "saga_donulmez",
    "sola_donulmez",
    "park_yeri",
    "park_etmek_yasaktir",
    "yaya_gecidi",
    "isikli_isaret_cihazi",
    "hiz_siniri_20",
    "hiz_siniri_30",
    "hiz_siniri_40",
    "hiz_siniri_50",
    "sagdan_gidiniz",
    "soldan_gidiniz",
    "saga_mecburi_yon",
    "sola_mecburi_yon",
    "ileri_mecburi_yon",
    "ileri_ve_saga_mecburi_yon",
    "ileri_ve_sola_mecburi_yon",
    "ileriden_saga_mecburi_yon",
    "ileriden_sola_mecburi_yon",
    "serit_duzenleme_levhasi_sag",
    "serit_duzenleme_levhasi_sol",
    "ada_etrafinda_donunuz",
    "iki_yonlu_yol",
    "tunel",
    "dikkat",
    "okul_gecidi",
    "yol_calismasi",
]


def require_blender():
    try:
        result = subprocess.run(
            ["blender", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("HATA: blender bulunamadı. Kurulum: sudo apt install -y blender")
        sys.exit(1)

    if result.returncode != 0:
        print(result.stdout)
        print("HATA: blender çalıştırılamadı.")
        sys.exit(1)


def write_blender_script(script_path: Path):
    script_path.write_text(
        r'''
import bpy
import sys
from pathlib import Path

argv = sys.argv
argv = argv[argv.index("--") + 1:]

sign_name = argv[0]
texture_path = Path(argv[1]).resolve()
fbx_path = Path(argv[2]).resolve()

# Sahneyi temizle
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Ölçüler metre kabul edilir.
# Tabela: 0.95 x 0.95 m, ince panel.
# Direk: 1.65 m.
board_w = 0.95
board_h = 0.95
board_t = 0.035
pole_h = 1.65
pole_r = 0.035

# Direk
bpy.ops.mesh.primitive_cylinder_add(
    vertices=24,
    radius=pole_r,
    depth=pole_h,
    location=(0.0, 0.0, pole_h / 2.0),
)
pole = bpy.context.object
pole.name = f"{sign_name}_pole"

mat_pole = bpy.data.materials.new(f"M_{sign_name}_pole")
mat_pole.diffuse_color = (0.12, 0.12, 0.12, 1.0)
pole.data.materials.append(mat_pole)

# Tabela paneli
# Panel merkezini direğin üstüne yakın koyuyoruz.
board_center_z = pole_h + board_h * 0.45

bpy.ops.mesh.primitive_cube_add(
    size=1.0,
    location=(0.0, -0.015, board_center_z),
)
board = bpy.context.object
board.name = f"{sign_name}_board"
board.dimensions = (board_w, board_t, board_h)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Texture material
mat = bpy.data.materials.new(f"M_{sign_name}_texture")
mat.use_nodes = True

nodes = mat.node_tree.nodes
bsdf = nodes.get("Principled BSDF")

tex_node = nodes.new(type="ShaderNodeTexImage")
tex_node.image = bpy.data.images.load(str(texture_path))
tex_node.extension = "CLIP"

mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.75

board.data.materials.append(mat)

# UV düzenle: bütün texture ön yüze otursun.
# Cube olduğu için default UV yeterli değil; hızlı çözüm olarak smart unwrap.
bpy.context.view_layer.objects.active = board
board.select_set(True)
pole.select_set(False)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
bpy.ops.object.mode_set(mode="OBJECT")

# Arka panel gri olsun diye ayrı ince arka plaka ekleyelim.
bpy.ops.mesh.primitive_cube_add(
    size=1.0,
    location=(0.0, 0.018, board_center_z),
)
back = bpy.context.object
back.name = f"{sign_name}_back"
back.dimensions = (board_w, board_t, board_h)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

mat_back = bpy.data.materials.new(f"M_{sign_name}_back")
mat_back.diffuse_color = (0.78, 0.78, 0.78, 1.0)
back.data.materials.append(mat_back)

# Objeleri seç ve tek FBX export et
bpy.ops.object.select_all(action="DESELECT")
pole.select_set(True)
board.select_set(True)
back.select_set(True)
bpy.context.view_layer.objects.active = board

fbx_path.parent.mkdir(parents=True, exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=str(fbx_path),
    use_selection=True,
    apply_unit_scale=True,
    bake_space_transform=False,
    object_types={"MESH"},
    path_mode="COPY",
    embed_textures=False,
    axis_forward="-Z",
    axis_up="Y",
)

print(f"EXPORTED {fbx_path}")
''',
        encoding="utf-8",
    )


def build_package():
    if not TEXTURE_DIR.exists():
        print(f"HATA: texture klasörü yok: {TEXTURE_DIR}")
        sys.exit(1)

    missing = []
    for sign in SIGNS:
        if not (TEXTURE_DIR / f"{sign}.png").exists():
            missing.append(sign)

    if missing:
        print("HATA: eksik texture dosyaları:")
        for item in missing:
            print(" -", item)
        sys.exit(1)

    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)

    PROPS_ROOT.mkdir(parents=True, exist_ok=True)

    blender_script = OUT_ROOT / "_make_one_sign_fbx.py"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_blender_script(blender_script)

    props_json = []

    for sign in SIGNS:
        prop_name = f"teknofest_sign_{sign}"
        prop_dir = PROPS_ROOT / prop_name
        prop_dir.mkdir(parents=True, exist_ok=True)

        src_texture = TEXTURE_DIR / f"{sign}.png"
        dst_texture = prop_dir / f"{prop_name}_Diff.png"
        shutil.copy2(src_texture, dst_texture)

        fbx_path = prop_dir / f"{prop_name}.fbx"

        cmd = [
            "blender",
            "--background",
            "--python",
            str(blender_script),
            "--",
            prop_name,
            str(dst_texture),
            str(fbx_path),
        ]

        print("FBX oluşturuluyor:", prop_name)
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print("HATA: Blender FBX export başarısız:", prop_name)
            sys.exit(result.returncode)

        props_json.append({
            "name": prop_name,
            "size": "small",
            "source": f"./Props/{prop_name}/{prop_name}.fbx",
            "tag": "TrafficSign",
        })

    package_data = {
        "maps": [],
        "props": props_json,
    }

    JSON_PATH.write_text(
        json.dumps(package_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("CARLA import paketi hazır:")
    print(PACKAGE_ROOT)
    print()
    print("JSON:")
    print(JSON_PATH)
    print()
    print("Beklenen blueprint id örneği:")
    print("static.prop.teknofest_sign_dur")
    print("static.prop.teknofest_sign_yol_ver")


def main():
    require_blender()
    build_package()


if __name__ == "__main__":
    main()
