# file: hypergraph_extractor/out_sem_2/mypykeen.py
from __future__ import annotations

__requires__ = ["Setuptools<81"]
import importlib_resources, importlib_metadata

from pykeen.triples import TriplesFactory
from pathlib import Path
import json, pprint, os
from pykeen.datasets import PathDataset, Dataset
from pykeen.pipeline import pipeline, replicate_pipeline_from_config, pipeline_from_path
from pykeen.training import training_loop
from pykeen.hpo import hpo_pipeline
from optuna.samplers import TPESampler
from pykeen.sampling import BasicNegativeSampler
from pykeen.predict import predict_target, predict_all
import pykeen
import torch
import numpy
import random
import pprint
import gc, re, pandas as pd
from pykeen.nn import TextRepresentation
from pykeen.models import TransD
torch.manual_seed(42)
random.seed(42)
numpy.random.seed(42)
#torch.device('cuda:0')
#torch.cpu.set_device('cpu')
#torch.cuda.set_device('cuda:0')

#torch.set_default_device('cuda:0')


#training_path = Path("pykeen_dataset/train/")
#testing_path = Path("pykeen_dataset/test/")
#validation_path = Path("pykeen_dataset/valid/")
#dataset  = PathDataset(training_path="pykeen_dataset/train/train.txt", testing_path="pykeen_dataset/test/test.txt", validation_path="pykeen_dataset/valid/valid.txt", create_inverse_triples=True)
with open("HRKG_merged/hrkg_triples.txt", "r") as fin, open("stripped_triples.txt", "w") as fout:
    for line in fin:
        h, r, t = line.rstrip("\n").split("\t")
        # keep only the predicate between the two colons
        g=re.search(r':([^:]+):', h)
        m = re.search(r':([^:]+):', r)
        y=re.search(r':([^:]+):', t)
        h_simple = g.group(1) if g else h
        r_simple = m.group(1) if m else r 
        t_simple = y.group(1) if y else t 
        fout.write(f"{h_simple}\t{r_simple}\t{t_simple}\n")
tf = TriplesFactory.from_path("HRKG_merged/hrkg_triples.txt", create_inverse_triples=True) # "hg_freebase.txt"
entity_representations = TextRepresentation.from_triples_factory(triples_factory=tf, encoder="transformer")
training, testing, validation = tf.split([.8, .1, .1], random_state=42)

print('causes' in tf.relation_to_id)
print("Alcohol" in tf.entity_to_id)

#output_fb= pykeen.triples.triples_factory.get_mapped_triples(x="hg_entities.tsv", triples=tf.triples, factory=TriplesFactory.from_path("hg_freebase.txt", create_inverse_triples=True))
#print(output_fb)
#output_fb.to_csv("output_fb.csv")

#""" um mure ermlp **proje ***transf ['autosf', 'boxe', 'compgcn', 'complex', 'complexliteral', 'conve', 'convkb', 'cooccurrencefiltered', 'cp', 'crosse', 'distma', 'distmult', 
#'distmultliteral', 'distmultliteralgated', 'ermlp', 'ermlpe', 'fixed', 'hole', 'inductivenodepiece', 'inductivenodepiecegnn', 'kg2e', 'mure', 'nodepiece', 
#'ntn', 'pairre', 'proje', 'quate', 'rescal', 'rgcn', 'rotate', 'se', 'simple', 'toruse', 'transd', 'transe', 'transf', 'transh', 'transr', 'tucker', 'um']"""

sampler = TPESampler(prior_weight=1.1, seed=42)

#result = hpo_pipeline( entity_representations=entity_representations,
result = hpo_pipeline(
    training=training,
    testing=testing,
    validation=validation,
    device='cuda',
    n_trials=25,
    n_jobs=1,
    sampler=sampler,
    model='transf',
    model_kwargs=dict( random_seed=42,       
        
        
    ),
    optimizer='Adam',
    training_loop='slcwa',
    training_kwargs=dict(pin_memory = False,drop_last=False,),
    #training_loop_kwargs=dict(drop_last=False,),
    negative_sampler="basic",
    negative_sampler_kwargs=dict(num_negs_per_pos=1),
    evaluator='RankBasedEvaluator',
    #clear_optimizer=True,
    #evaluation_fallback=False,
    epochs=25,
)
print(result)
result.save_to_directory('pykeen_results_fb')

raw_cfg = result._get_best_study_config()         # dict with "pipeline"
cfg     = raw_cfg["pipeline"]

def scrub_placeholders(obj):
    """Recursively replace any '<...>' string with None."""
    if isinstance(obj, dict):
        return {k: scrub_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_placeholders(v) for v in obj]
    if isinstance(obj, str) and obj.startswith('<'):
        return None
    return obj

clean_cfg = scrub_placeholders(raw_cfg)           # whole dict (meta+pipeline)

# ---------------------------------------------------------------
# 3) Re-run once with replicate_pipeline_from_config
# ---------------------------------------------------------------
out_dir = Path("pykeen_runs_fb/best")
out_dir.mkdir(exist_ok=True, parents=True)
dataset = Dataset.from_tf(tf)
best_pipeline_result = replicate_pipeline_from_config(
    config      = clean_cfg,
    dataset=dataset,
    directory   = out_dir,
    replicates  = 1,
    device      = "cuda",
    move_to_cpu = True,
    save_replicates=True,          # keep it light-weight
)

model = torch.load(f'{out_dir}/replicates/replicate-00000/trained_model.pkl', weights_only=False)

factory = tf 


print('causes' in factory.relation_to_id)
print("Alcohol" in factory.entity_to_id) #'/m/n147'

def all_causes_predictions(model, factory, head, relation):
    # every key that contains ":causes:"
    cause_rels = [r for r in factory.relation_to_id if f':{relation}:' in r]
    
    dfs = []
    for r in cause_rels:
        tp = predict_target(model=model,
                            relation=r,
                            head=head,
                            triples_factory=factory)
        df = tp.df
        df['relation'] = r          # keep track of which variant it came from
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

df = all_causes_predictions(model, factory, head="Alcohol", relation="causes")
df.to_csv("df_fb.csv")
print(df.head())

#df = predict_target(
#        model=model,
#        relation="causes", #relation
#        head="Alcohol", #"/m/n147"
#        triples_factory=factory,
#)
#print(df)
#df.df.to_csv("df_fb.csv")

#df_all_head= predict_all(
#        model=model,
#        k=None,
#        batch_size=None,
#        mode="testing",
#        target="head",
#)
#df_all_head.df.to_csv("df_all_head_fb.csv")


#df_all_relation= predict_all(
#        model=model,
#        k=None,
#        batch_size=None,
#        mode="testing",
#        target="relation",
#)
#df_all_relation.df.to_csv("df_all_relation_fb.csv")

#df_all_tail= predict_all(
#        model=model,
#        k=None,
#        batch_size=None,
#        mode="testing",
#        target="tail",
#)
#df_all_tail.df.to_csv("df_all_tail_fb.csv")


gc.collect()
torch.cuda.empty_cache()
gc.collect()
