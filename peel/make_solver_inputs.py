import torch
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pk=torch.load(PEEL+"peel_committee.pt",map_location="cpu",weights_only=False)
assert "teacher_state" in pk
out={"dims":pk["dims"],"pop_states":[{k:v for k,v in sd.items()} for sd in pk["pop_states"]]}  # NO teacher_state
torch.save(out, PEEL+"solver_inputs.pt")
# sanity: ensure no truth leaked
assert "teacher_state" not in out and all("teacher" not in str(k) for sd in out["pop_states"] for k in sd)
print("wrote solver_inputs.pt with", len(out["pop_states"]), "committee members, dims", out["dims"], "-- NO teacher_state")
