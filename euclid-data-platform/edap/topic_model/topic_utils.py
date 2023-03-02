import networkx as nx
import pandas as pd
import json


class TopicModelParser():
    # json topic model
    # graph형태로 변환
    def __init__(self, topic_model):
        self.topic_model = json.loads(topic_model)
        self.graph = nx.DiGraph()
        self.synonym_word = pd.DataFrame()
        self.rel_word = pd.DataFrame()
        self.node_list = []
        self.edge_list = []
        self._make_node_edge_list()

    def _make_relationship(self, children_list, parent_id=None, current_depth=1):

        for child in children_list:

            if not 'check' in child or child['check'] is None or child['check'] == 'N':
                continue

            _check = child['check']
            _id = child['id']
            _name = child['name']
            _relation = None

            if 'relation' in child:
                _relation = child['relation']

            if current_depth == 1:
                self.node_list.append((_id, {'name': _name, 'node_type': 'topic'}))
            elif current_depth == 2:
                self.node_list.append((_id, {'name': _name, 'node_type': 'word'}))
            elif current_depth > 1:
                self.node_list.append((_id, {'name': _name, 'node_type': 'r_word'}))

            sub_children = child['children']
            if parent_id is not None and _check != 'N':
                edge = (parent_id, _id, {})
                if _check == 'Y':
                    edge[2].update({'op': 'OR'})
                elif _check == 'E':
                    edge[2].update({'op': 'NOT'})
                if current_depth > 2:
                    edge[2].update({'wr': _relation})
                self.edge_list.append(edge)
            if sub_children is not None and len(sub_children) > 0:
                self._make_relationship(sub_children, _id, current_depth + 1)

    def _make_node_edge_list(self):

        if 'check' in self.topic_model or self.topic_model['check'] == 'Y' or 'children' in self.topic_model:
            _kwd = self.topic_model['name']
            _kwd_id = self.topic_model['id']
            topics = self.topic_model['children']

            self.node_list.append((_kwd_id, {'name': _kwd, 'node_type': 'keyword'}))
            self._make_relationship(topics, _kwd_id)

    def figure_relation_df(self):
        # self.gp => "syno", "relatedword"
        self.graph.add_nodes_from(self.node_list)
        self.graph.add_edges_from(self.edge_list)

        gp_wr_data = self.graph.edges.data('wr')
        rel_edge_list = [(self.graph.nodes[n1].get('name'), self.graph.nodes[n2].get('name'), _type) for (n1, n2, _type)
                         in gp_wr_data if _type in ['R', 'S']]
        #         rel_edge_list = [(n2,n1,_type) if [n1,n2] != sorted([n1,n2]) else (n1,n2,_type) for (n1,n2,_type) in rel_edge_list]

        return pd.DataFrame(rel_edge_list, columns=["main_term", "sub_term", "relation"])

    def export_synonyms(self):
        sr_word_df = self.figure_relation_df()
        target_df = sr_word_df.loc[sr_word_df.relation == 'S'][["main_term", "sub_term"]]
        self.synonym_word = list(target_df.to_records(index=False))

        return self.synonym_word

    def export_relations(self):
        sr_word_df = self.figure_relation_df()
        target_df = sr_word_df.loc[sr_word_df.relation == 'R'][["main_term", "sub_term"]]
        self.synonym_word = list(target_df.to_records(index=False))

        return self.synonym_word

    def _create_search_query(self):
        # 검색식 만들기
        return ''