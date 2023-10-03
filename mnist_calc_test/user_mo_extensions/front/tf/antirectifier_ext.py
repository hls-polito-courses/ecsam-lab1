from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op

class antirectifierFrontExtractor(FrontExtractorOp):
	op = 'Antirectifier' 
	enabled = True

	@staticmethod
	def extract(node):
		proto_layer = node.pb
		# extracting parameters from TensorFlow layer and prepare them for IR
		param = proto_layer.attr
		attrs = { 
			'op': __class__.op 
		}
		# update the attributes of the node
		Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)
		return __class__.enabled
