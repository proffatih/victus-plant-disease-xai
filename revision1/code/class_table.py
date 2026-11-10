"""REV1 (R1-2, R1-1): biological description of the evaluated classes.

One row per PlantVillage class: host crop, disease, causal agent and agent type,
and the number of images in the PlantVillage train/test partitions, the PlantDoc
shared subset (all images; official train/test split) and the PlantWild shared
subset, together with the source folder names that were mapped onto it. Writes
results/class_table.csv and the LaTeX body of the supplementary-ready table.
Causal agents follow standard plant-pathology references for each disease; the
same disease name is assumed to denote the same agent across datasets, which is
exactly the assumption the mapping makes and which the table exposes.
"""
from __future__ import annotations
import csv, json, os
from collections import Counter, defaultdict
from pathlib import Path

R3 = Path(__file__).resolve().parents[2] / "R3" / "results"
OUT = Path(__file__).resolve().parents[1] / "results"

# class -> (host, disease, causal agent, agent type)
BIO = {
 "Apple___Apple_scab": ("Apple", "Apple scab", "Venturia inaequalis", "Fungus"),
 "Apple___Black_rot": ("Apple", "Black rot", "Diplodia seriata (Botryosphaeria obtusa)", "Fungus"),
 "Apple___Cedar_apple_rust": ("Apple", "Cedar apple rust", "Gymnosporangium juniperi-virginianae", "Fungus"),
 "Apple___healthy": ("Apple", "Healthy", "--", "--"),
 "Blueberry___healthy": ("Blueberry", "Healthy", "--", "--"),
 "Cherry___Powdery_mildew": ("Cherry", "Powdery mildew", "Podosphaera clandestina", "Fungus"),
 "Cherry___healthy": ("Cherry", "Healthy", "--", "--"),
 "Corn___Cercospora_leaf_spot Gray_leaf_spot": ("Maize", "Grey leaf spot", "Cercospora zeae-maydis", "Fungus"),
 "Corn___Common_rust": ("Maize", "Common rust", "Puccinia sorghi", "Fungus"),
 "Corn___Northern_Leaf_Blight": ("Maize", "Northern leaf blight", "Exserohilum turcicum", "Fungus"),
 "Corn___healthy": ("Maize", "Healthy", "--", "--"),
 "Grape___Black_rot": ("Grape", "Black rot", "Phyllosticta ampelicida (Guignardia bidwellii)", "Fungus"),
 "Grape___Esca_(Black_Measles)": ("Grape", "Esca (black measles)", "Phaeomoniella chlamydospora and others", "Fungal complex"),
 "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Grape", "Isariopsis leaf spot", "Pseudocercospora vitis", "Fungus"),
 "Grape___healthy": ("Grape", "Healthy", "--", "--"),
 "Orange___Haunglongbing_(Citrus_greening)": ("Orange", "Huanglongbing (citrus greening)", "Candidatus Liberibacter asiaticus", "Bacterium"),
 "Peach___Bacterial_spot": ("Peach", "Bacterial spot", "Xanthomonas arboricola pv. pruni", "Bacterium"),
 "Peach___healthy": ("Peach", "Healthy", "--", "--"),
 "Pepper,_bell___Bacterial_spot": ("Bell pepper", "Bacterial spot", "Xanthomonas spp.", "Bacterium"),
 "Pepper,_bell___healthy": ("Bell pepper", "Healthy", "--", "--"),
 "Potato___Early_blight": ("Potato", "Early blight", "Alternaria solani", "Fungus"),
 "Potato___Late_blight": ("Potato", "Late blight", "Phytophthora infestans", "Oomycete"),
 "Potato___healthy": ("Potato", "Healthy", "--", "--"),
 "Raspberry___healthy": ("Raspberry", "Healthy", "--", "--"),
 "Soybean___healthy": ("Soybean", "Healthy", "--", "--"),
 "Squash___Powdery_mildew": ("Squash", "Powdery mildew", "Podosphaera xanthii", "Fungus"),
 "Strawberry___Leaf_scorch": ("Strawberry", "Leaf scorch", "Diplocarpon earlianum", "Fungus"),
 "Strawberry___healthy": ("Strawberry", "Healthy", "--", "--"),
 "Tomato___Bacterial_spot": ("Tomato", "Bacterial spot", "Xanthomonas spp.", "Bacterium"),
 "Tomato___Early_blight": ("Tomato", "Early blight", "Alternaria solani / A. linariae", "Fungus"),
 "Tomato___Late_blight": ("Tomato", "Late blight", "Phytophthora infestans", "Oomycete"),
 "Tomato___Leaf_Mold": ("Tomato", "Leaf mould", "Fulvia fulva (Passalora fulva)", "Fungus"),
 "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria leaf spot", "Septoria lycopersici", "Fungus"),
 "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Two-spotted spider mite", "Tetranychus urticae", "Arthropod pest"),
 "Tomato___Target_Spot": ("Tomato", "Target spot", "Corynespora cassiicola", "Fungus"),
 "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow leaf curl", "Tomato yellow leaf curl virus", "Virus"),
 "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic", "Tomato mosaic virus", "Virus"),
 "Tomato___healthy": ("Tomato", "Healthy", "--", "--"),
}


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pv_tr = Counter(r["class"] for r in read_csv(R3 / "split_train.csv"))
    pv_te = Counter(r["class"] for r in read_csv(R3 / "split_test.csv"))
    pd_rows = read_csv(R3 / "plantdoc_shared.csv")
    pd_all = Counter(r["class"] for r in pd_rows)
    pd_test = Counter(r["class"] for r in pd_rows if "/test/" in r["path"])
    pd_src = defaultdict(set)
    for r in pd_rows:
        pd_src[r["class"]].add(Path(r["path"]).parent.name)
    pw = json.load(open(R3 / "plantwild_mapping.json"))
    pw_src = defaultdict(list)
    for src, dst in pw["mapping"].items():
        pw_src[dst].append(src)
    pw_split = read_csv(R3 / "plantwild_split_42.csv")
    pw_n = Counter(r["class"] for r in pw_split)
    assert set(pv_tr) == set(BIO), set(pv_tr) ^ set(BIO)

    rows = []
    for c in sorted(BIO):
        host, dis, agent, kind = BIO[c]
        rows.append({"class": c, "host": host, "disease": dis, "agent": agent, "agent_type": kind,
                     "pv_train": pv_tr[c], "pv_test": pv_te[c],
                     "plantdoc_all": pd_all.get(c, 0), "plantdoc_test": pd_test.get(c, 0),
                     "plantwild": pw_n.get(c, 0),
                     "plantdoc_source": "; ".join(sorted(pd_src.get(c, []))),
                     "plantwild_source": "; ".join(sorted(pw_src.get(c, [])))})
    with open(OUT / "class_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # summary
    shared_pd = [r for r in rows if r["plantdoc_all"]]
    shared_pw = [r for r in rows if r["plantwild"]]
    summ = {"n_classes": len(rows), "n_hosts": len({r["host"] for r in rows}),
            "pd_classes": len(shared_pd), "pd_hosts": len({r["host"] for r in shared_pd}),
            "pw_classes": len(shared_pw), "pw_hosts": len({r["host"] for r in shared_pw}),
            "agent_types_pd": Counter(r["agent_type"] for r in shared_pd),
            "agent_types_pw": Counter(r["agent_type"] for r in shared_pw),
            "pd_images": sum(r["plantdoc_all"] for r in rows),
            "pw_images": sum(r["plantwild"] for r in rows),
            "pd_min_max": (min(r["plantdoc_all"] for r in shared_pd), max(r["plantdoc_all"] for r in shared_pd)),
            "pw_min_max": (min(r["plantwild"] for r in shared_pw), max(r["plantwild"] for r in shared_pw)),
            "pv_train_min_max": (min(pv_tr.values()), max(pv_tr.values())),
            "in_both_field": len([r for r in rows if r["plantdoc_all"] and r["plantwild"]])}
    json.dump(summ, open(OUT / "class_table_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # LaTeX table
    TEX = {
     "Venturia inaequalis": r"\textit{Venturia inaequalis}",
     "Diplodia seriata (Botryosphaeria obtusa)": r"\textit{Diplodia seriata}",
     "Gymnosporangium juniperi-virginianae": r"\textit{Gymnosporangium juniperi-virginianae}",
     "Podosphaera clandestina": r"\textit{Podosphaera clandestina}",
     "Cercospora zeae-maydis": r"\textit{Cercospora zeae-maydis}",
     "Puccinia sorghi": r"\textit{Puccinia sorghi}",
     "Exserohilum turcicum": r"\textit{Exserohilum turcicum}",
     "Phyllosticta ampelicida (Guignardia bidwellii)": r"\textit{Phyllosticta ampelicida}",
     "Phaeomoniella chlamydospora and others": r"\textit{Phaeomoniella chlamydospora} and others",
     "Pseudocercospora vitis": r"\textit{Pseudocercospora vitis}",
     "Candidatus Liberibacter asiaticus": r"`\textit{Candidatus} Liberibacter asiaticus'",
     "Xanthomonas arboricola pv. pruni": r"\textit{Xanthomonas arboricola} pv.~\textit{pruni}",
     "Xanthomonas spp.": r"\textit{Xanthomonas} spp.",
     "Alternaria solani": r"\textit{Alternaria solani}",
     "Phytophthora infestans": r"\textit{Phytophthora infestans}",
     "Podosphaera xanthii": r"\textit{Podosphaera xanthii}",
     "Diplocarpon earlianum": r"\textit{Diplocarpon earlianum}",
     "Alternaria solani / A. linariae": r"\textit{Alternaria solani}, \textit{A.~linariae}",
     "Fulvia fulva (Passalora fulva)": r"\textit{Fulvia fulva}",
     "Septoria lycopersici": r"\textit{Septoria lycopersici}",
     "Tetranychus urticae": r"\textit{Tetranychus urticae}",
     "Corynespora cassiicola": r"\textit{Corynespora cassiicola}",
     "Tomato yellow leaf curl virus": "Tomato yellow leaf curl virus",
     "Tomato mosaic virus": "Tomato mosaic virus",
     "--": "--"}
    body = []
    for r in sorted(rows, key=lambda r: (r["host"], r["disease"] != "Healthy", r["disease"])):
        body.append(f"{r['host']} & {r['disease']} & {TEX[r['agent']]} & {r['agent_type']} & "
                    f"{r['pv_train']:,} & {r['pv_test']} & {r['plantdoc_all'] or '--'} & "
                    f"{r['plantwild'] or '--'} \\\\".replace(",", "\\,"))
    tot = (f"\\midrule\nTotal & & & & {sum(r['pv_train'] for r in rows):,} & {sum(r['pv_test'] for r in rows):,} & "
           f"{summ['pd_images']:,} & {summ['pw_images']:,} \\\\").replace(",", "\\,")
    (OUT / "tab_classes_body.tex").write_text("\n".join(body) + "\n" + tot + "\n")


if __name__ == "__main__":
    main()
