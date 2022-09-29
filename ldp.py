from ply.lex import lex
from ply.yacc import yacc
from rdflib.term import URIRef
from rdflib.namespace import RDF, RDFS, OWL
# --- Tokenizer

# All tokens must be named in advance.
tokens = ('LOG', 'URI', 'LPAREN', 'RPAREN')

# Ignored characters
t_ignore = ' \t'

# Token matching rules are written as regexs
t_LOG = r'\w+'
t_URI = r'<[^>]+>'
t_LPAREN = r'\('
t_RPAREN = r'\)'

# Ignored token with an action associated with it
def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')


# Error handler for illegal characters
def t_error(t):
    t.lexer.skip(1)


# Build the lexer object
lexer = lex()


# --- Parser

# Write functions for each grammar rule which is
# specified in the docstring.

tm = {
    'ObjectIntersectionOf': OWL.intersectionOf,
    'ObjectSomeValuesFrom': OWL.someValuesFrom,
    'ObjectUnionOf': OWL.unionOf
}
def p_tree(p):
    '''
    tree : LOG LPAREN tree RPAREN tree
        | URI tree
        | empty
    '''

    if len(p) == 3:
        if p[2] is None:
            p[2] = []
        if type(p[2]) is not list:
            p[2] = [p[2]]

        p[2].insert(0, URIRef(p[1][1:-1]))

        p[0] = p[2]
    elif len(p) == 6:
        if p[5] is None:
            p[5] = []
        if p[3] is None:
            p[3] = []

        if type(p[3]) is not list:
            p[3] = [p[3]]

        if type(p[5]) is not list:
            p[5] = [p[5]]



        ch = p[3] + p[5]
        p[0] = (tm[p[1]], ch)



def p_empty(p):
    '''

    empty :
    '''




def p_error(p):
    raise Exception(f'Syntax error at {p.value!r}')


parser = yacc()
