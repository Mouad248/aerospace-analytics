import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .core import train,rotation_and_optimization

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','scenario','serve']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--port',type=int,default=8060)
    a=p.parse_args();root=a.root.resolve()
    if a.command=='train':train(root)
    elif a.command=='scenario':print(rotation_and_optimization(root))
    else:
        print(f'http://127.0.0.1:{a.port}/reports/latest/',flush=True)
        ThreadingHTTPServer(('127.0.0.1',a.port),partial(SimpleHTTPRequestHandler,directory=str(root))).serve_forever()

if __name__=='__main__':main()
