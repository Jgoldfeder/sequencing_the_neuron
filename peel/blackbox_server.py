import socket, io, os, sys, torch
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
SOCK="/tmp/claude-1001/-home-judah/2cb1357d-51f9-4a19-9d02-8f39bb198f3b/scratchpad/bb.sock"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"; dev="cuda"
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)   # teacher lives ONLY here
net=MLP(pk["dims"],act="sigmoid").to(dev).double(); net.load_state_dict(pk["teacher_state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
def recvall(c,n):
    b=b""
    while len(b)<n:
        d=c.recv(n-len(b))
        if not d: return None
        b+=d
    return b
def recv_msg(c):
    h=recvall(c,8)
    if h is None: return None
    n=int.from_bytes(h,"big"); return torch.load(io.BytesIO(recvall(c,n)),weights_only=False)
def send_msg(c,o):
    buf=io.BytesIO(); torch.save(o,buf); d=buf.getvalue(); c.sendall(len(d).to_bytes(8,"big")+d)
if os.path.exists(SOCK): os.remove(SOCK)
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(SOCK); s.listen(1)
print("BBSERVER READY",flush=True)
conn,_=s.accept(); nq=0
while True:
    m=recv_msg(conn)
    if m is None or m.get("cmd")=="stop": break
    x=m["x"].to(dev).double(); nq+=x.shape[0]
    with torch.no_grad(): out=net(x).detach().cpu()
    send_msg(conn,{"out":out})
print(f"BBSERVER done, served {nq} rows",flush=True); conn.close(); s.close()
