import numpy as np
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.front.extractor import FrontExtractorOp
from mo.front.common.partial_infer.elemental import copy_shape_infer

class AntirectifierOp(Op):
	op = 'Antirectifier'
	def __init__(self, graph, attrs):
		mandatory_props = dict(
			type=__class__.op,
			op=__class__.op,
			infer=AntirectifierOp.infer
		)
		super().__init__(graph, mandatory_props, attrs)
	@staticmethod
	def infer(node: Node):
		outn = node.out_node(0)
		inn = node.in_node(0)
		outn.shape = np.copy(inn.shape)
		outn.shape[:-1] = inn.shape[:-1]
		outn.shape[-1] = 2*(inn.shape[-1])
