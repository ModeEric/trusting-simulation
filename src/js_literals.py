"""Narrow parser for pinned website data literals. No JavaScript execution.

Accepts only numbers, JSON double-quoted strings, arrays, objects, arithmetic
on numeric literals, and a(score, successRate). Unknown syntax fails closed.
"""
import json,re
TOKEN=re.compile(r'\s*("(?:\\.|[^"\\])*"|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_$][\w$]*|[][{}():,+*/-])')
class Parser:
    def __init__(self,text):self.text=text;self.pos=0;self.token=None;self.next()
    def next(self):
        m=TOKEN.match(self.text,self.pos)
        if not m:self.token=None;return
        self.token=m.group(1);self.pos=m.end()
    def take(self,t):
        if self.token!=t:raise ValueError(f'Expected {t}, got {self.token}')
        self.next()
    def value(self):
        t=self.token
        if t=='[':
            self.next();out=[]
            while self.token!=']':
                out.append(self.value())
                if self.token!=',':break
                self.next()
            self.take(']');return out
        if t=='{':
            self.next();out={}
            while self.token!='}':
                k=self.token
                if k is None:raise ValueError('Truncated object')
                key=json.loads(k) if k.startswith('"') else k
                self.next();self.take(':');out[key]=self.value()
                if self.token!=',':break
                self.next()
            self.take('}');return out
        if t and t.startswith('"'):
            self.next();return json.loads(t)
        if t=='a':
            self.next();self.take('(');score=self.number();self.take(',');sr=self.number();self.take(')')
            return {'score':score,'successRate':sr}
        return self.number()
    def number(self):
        sign=1
        if self.token=='-':sign=-1;self.next()
        if self.token is None or not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',self.token):
            raise ValueError(f'Unsupported literal {self.token}')
        x=sign*float(self.token);self.next()
        while self.token in {'*','/'}:
            op=self.token;self.next()
            if self.token is None or not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',self.token):raise ValueError('Nonliteral arithmetic')
            y=float(self.token);self.next();x=x*y if op=='*' else x/y
        return x

def assigned_literal(text,name):
    m=re.search(r'\b'+re.escape(name)+r'=',text)
    if not m:raise ValueError(f'Missing data variable: {name}')
    return Parser(text[m.end():]).value()
