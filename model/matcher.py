import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from einops import rearrange

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def extract_valid_data(data, valid):
    bs, seq, q, dim = data.shape
    valid = valid.squeeze(-1)  # (bs, seq)

    mask = torch.arange(q, device=valid.device).expand(bs, seq, q) < valid.unsqueeze(-1)  # (bs, seq, q)

    valid_data = data.masked_select(mask.unsqueeze(-1))
    valid_data = valid_data.view(-1, dim)  #(n, dim)

    return valid_data


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, seq_len, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, seq_len, num_queries, 4] with the predicted box coordinates

            targets: This is a dict that contains at least these entries:
                 "hand_type": Tensor of dim [batch_size, seq_len, num_queries] containing the class labels
                 "boxes": Tensor of dim [batch_size, seq_len, num_queries, 4] containing the target box coordinates
                 "valid": Tensor of dim [batch_size, seq_len, 1] showing the first n from num_queries samples are valid.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, seq_len, num_queries = outputs["pred_logits"].shape[:3]

        # We flatten to compute the cost matrices in a batch
        out_prob = rearrange(outputs["pred_logits"], "bs seq q cls -> (bs seq q) cls")  # [batch_size * seq_len * num_queries, num_classes]
        out_bbox = rearrange(outputs["pred_boxes"], "bs seq q box -> (bs seq q) box")  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        tgt_ids = extract_valid_data(targets["hand_type"][:, :, :, None], targets["valid"]).squeeze(-1)
        tgt_bbox = extract_valid_data(targets["boxes"], targets["valid"])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = rearrange(C, "(bs seq q) size -> (bs seq) q size", bs=bs, seq=seq_len, q=num_queries).cpu()
        # C = C.view(bs, seq_len, num_queries, -1).cpu()

        # sizes = [len(v["boxes"]) for v in targets]
        sizes = rearrange(targets['valid'], "bs seq c -> (bs seq c)").tolist()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]