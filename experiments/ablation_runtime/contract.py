"""Input-branch contract retained from the original final ablation suite."""
VARIANTS = ('FULL', 'NO_CONDITION', 'NO_QUERY', 'NO_PRIOR',
            'ONLY_CONDITION', 'ONLY_QUERY', 'ONLY_PRIOR')
ROLE_MAP = {
    'FULL': (True, True, True), 'NO_CONDITION': (False, True, True),
    'NO_QUERY': (True, False, True), 'NO_PRIOR': (True, True, False),
    'ONLY_CONDITION': (True, False, False), 'ONLY_QUERY': (False, True, False),
    'ONLY_PRIOR': (False, False, True),
}

def roles(variant):
    return ROLE_MAP[variant]
