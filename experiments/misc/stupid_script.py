import fusion_tasks

fuses_names = [
    "GUE_human_tfs",
    "GUE_mouse_tfs",
    "GUE_mouse_tfs",
    "GUE_all_tfs",
    "GUE_human_promoters",
    "GUE_yeast_EMP",
    "GUE_human_splicing",
    "GUE_tfs_promoters_splicing",
    "GUE_all",
]
fuses = [
    fusion_tasks.GUE_human_tfs,
    fusion_tasks.GUE_mouse_tfs,
    fusion_tasks.GUE_mouse_tfs,
    fusion_tasks.GUE_all_tfs,
    fusion_tasks.GUE_human_promoters,
    fusion_tasks.GUE_yeast_EMP,
    fusion_tasks.GUE_human_splicing,
    fusion_tasks.GUE_tfs_promoters_splicing,
    fusion_tasks.GUE_all,
]
for i in range(len(fuses)):
    fuse = fuses[i]
    names = []
    for adapter_name in fuses[i]:
        names.append(f"./adapters/DNABERT_{adapter_name}_adapter/{adapter_name}")
    with open("fusion_addresses.py", "a") as f:
        f.write(f"{fuses_names[i]} = {names}")
        f.write("\n")
