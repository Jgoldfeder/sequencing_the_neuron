from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, CDTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "cd": CDTrainer
}
