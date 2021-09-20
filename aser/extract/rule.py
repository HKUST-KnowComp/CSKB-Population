# CONNECTIVE_LIST = frozenset(['before', 'afterward', 'next', 'then', 'till', 'until', 'after', 'earlier', 'once', 'previously',
#                    'meantime', 'meanwhile', 'simultaneously', 'because', 'for', 'accordingly', 'consequently', 'so',
#                    'thus', 'therefore', 'if', 'but', 'conversely', 'however', 'nonetheless', 'although', 'and',
#                    'additionally', 'also', 'besides', 'further', 'furthermore', 'similarly', 'likewise', 'moreover',
#                    'plus', 'specifically', 'alternatively', 'or', 'otherwise', 'unless', 'instead', 'except'])

CLAUSE_WORDS = frozenset(['when', 'who', 'what', 'where', 'how', 'When', 'Who', 'What', 'Where', 'How', 'why', 'Why', 'which',
                          'Which', '?'])


class Rule:
    def __init__(self, rules):
        if rules is None:
            self.positive_rules = list()
            self.negative_rules = list()
        else:
            self.positive_rules = rules['positive_rules']
            self.negative_rules = rules['negative_rules']


class EventualityRule(object):
    def __init__(self):
        self.positive_rules = list()
        self.possible_rules = list()
        self.negative_rules = list()


EVENTUALITY_PATTERNS = ['s-v', 's-v-o', 's-v-a', 's-v-o-o', 's-be-a', 's-be-o', 's-v-be-a', 's-v-be-o', 's-v-v-o',
                        's-v-v', 'spass-v', 's-v-o-v-o', 's-v-o-be-a', 's-v-o-be-o', 'spass-v-v-o', 'spass-v-o',
                        'there-be-o', 's-v-o-v-o-o']

ALL_EVENTUALITY_RULES = dict()

# nsubj-verb
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('S1',
                              '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case/compound:prt',
                              'NA1'))
E_rule.possible_rules.append(('V1',
                              '+advmod/neg/aux/compound:prt/mark',
                              'NA2'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA3'))
ALL_EVENTUALITY_RULES['s-v'] = E_rule


# subj-verb-dobj
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'dobj', 'O1'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/aux/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-v-o'] = E_rule


# subj-verb-dobj1-dobj2
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'dobj', 'O1'))
E_rule.positive_rules.append(('V1', 'iobj', 'O2'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('O2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O2M'))
E_rule.possible_rules.append(('O2M', 'case', 'O2MP'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O2', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-v-o-o'] = E_rule


# subj-be-obj
E_rule = EventualityRule()
E_rule.positive_rules.append(('A1', '^cop', 'V1'))
E_rule.positive_rules.append(('A1', 'nsubj', 'S1'))
E_rule.possible_rules.append(('A1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'A1M'))
E_rule.possible_rules.append(('A1M', 'case', 'A1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('A1', '+acl/advmod/neg/aux/compound:prt/det/amod/compound/nmod:poss/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('A1',
                              """-ccomp/parataxis/mark/nmod:npmod/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-be-a'] = E_rule


# subj-verb-be-object
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'A1'))
E_rule.positive_rules.append(('A1', 'cop', 'V2'))
E_rule.possible_rules.append(('A1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'A1M'))
E_rule.possible_rules.append(('A1M', 'case', 'A1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
E_rule.possible_rules.append(('A1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-v-be-a'] = E_rule


# subj-verb-verb-object
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
E_rule.positive_rules.append(('V2', 'dobj', 'O1'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V2M'))
E_rule.possible_rules.append(('V2M', 'case', 'V2MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
E_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt/mark', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V2',
                              """-ccomp/parataxis/nsubj:xsubj/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V0', '^ccomp', 'V1'))
E_rule.negative_rules.append(('V0', '^ccomp', 'V2'))
ALL_EVENTUALITY_RULES['s-v-v-o'] = E_rule


# subj-verb-verb
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
E_rule.possible_rules.append(('V2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V2M'))
E_rule.possible_rules.append(('V2M', 'case', 'V2MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
E_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt/mark', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V2',
                              """-ccomp/parataxis/nsubj:xsubj/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
# E_rule.negative_rules.append(('V0', '^ccomp', 'V1'))
# E_rule.negative_rules.append(('V0', '^ccomp', 'V2'))
ALL_EVENTUALITY_RULES['s-v-v'] = E_rule


# nsubjpass-verb
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubjpass', 'S1'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/auxpass/compound:prt/mark', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['spass-v'] = E_rule


# subj-verb-dobj1-verb-dobj2
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'dobj', 'O1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
E_rule.positive_rules.append(('V2', 'dobj', 'O2'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V2M'))
E_rule.possible_rules.append(('V2M', 'case', 'V2MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('O2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O2M'))
E_rule.possible_rules.append(('O2M', 'case', 'O2MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O2', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V2',
                              """-ccomp//nsubj:xsubj/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-v-o-v-o'] = E_rule

# nsubj-verb-dobj1-verb-dobj2
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'dobj', 'O1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'A1'))
E_rule.positive_rules.append(('A1', 'cop', 'V2'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V2M'))
E_rule.possible_rules.append(('V2M', 'case', 'V2MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('O2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O2M'))
E_rule.possible_rules.append(('O2M', 'case', 'O2MP'))
E_rule.possible_rules.append(('A1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'A1M'))
E_rule.possible_rules.append(('A1M', 'case', 'A1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('A1', '+acl/advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V2',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-v-o-be-a'] = E_rule


# nsbjpass-verb-verb-dobj
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubjpass', 'S1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
E_rule.positive_rules.append(('V2', 'dobj', 'O1'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V2M'))
E_rule.possible_rules.append(('V2M', 'case', 'V2MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/auxpass', 'NA'))
E_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V2',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['spass-v-v-o'] = E_rule

# nsubj-verb-dobj
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubjpass', 'S1'))
E_rule.positive_rules.append(('V1', 'dobj', 'O1'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/auxpass', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['spass-v-o'] = E_rule

# there-be-obj
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'expl', 'ex1'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/auxpass', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['there-be-o'] = E_rule

# nsubj-verb1-dobj1-verb2-dobj2-iobj3
E_rule = EventualityRule()
E_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
E_rule.positive_rules.append(('V1', 'dobj', 'O1'))
E_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
E_rule.positive_rules.append(('V2', 'dobj', 'O2'))
E_rule.positive_rules.append(('V2', 'iobj', 'O3'))
E_rule.possible_rules.append(('S1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'S1M'))
E_rule.possible_rules.append(('S1M', 'case', 'S1MP'))
E_rule.possible_rules.append(('V1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V1M'))
E_rule.possible_rules.append(('V1M', 'case', 'V1MP'))
E_rule.possible_rules.append(('V2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'V2M'))
E_rule.possible_rules.append(('V2M', 'case', 'V2MP'))
E_rule.possible_rules.append(('O1',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O1M'))
E_rule.possible_rules.append(('O1M', 'case', 'O1MP'))
E_rule.possible_rules.append(('O2',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O2M'))
E_rule.possible_rules.append(('O2M', 'case', 'O2MP'))
E_rule.possible_rules.append(('O3',
                              '+nmod:near/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about/nmod:agent',
                              'O3M'))
E_rule.possible_rules.append(('O3M', 'case', 'O3MP'))
E_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt', 'NA'))
E_rule.possible_rules.append(('S1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O1', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O2', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.possible_rules.append(('O3', '+acl/amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
E_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
E_rule.negative_rules.append(('V2',
                              """-ccomp//nsubj:xsubj/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
ALL_EVENTUALITY_RULES['s-v-o-v-o-o'] = E_rule


SEED_CONNECTIVE_DICT = {
    'Precedence': [['before']],
    'Succession': [['after']],
    'Synchronous': [['meanwhile'], ['at', 'the', 'same', 'time']],
    'Reason': [['because']],
    'Result': [['so'], ['thus'], ['therefore']],
    'Condition': [['if']],
    'Contrast': [['but'], ['however']],
    'Concession': [['although']],
    'Conjunction': [['and'], ['also']],
    'Instantiation': [['for', 'example'], ['for', 'instance']],
    'Restatement': [['in', 'other', 'words']],
    'Alternative': [['or'], ['unless']],
    'ChosenAlternative': [['instead']],
    'Exception': [['except']],
    'Co_Occurrence': list()
}
