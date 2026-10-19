"""Knowledge-distillation starter.

Expected workflow:
1. Train the 10-class Pest-ConFormer teacher.
2. Train the selected student normally.
3. Run this script logic for the same student with KD.

This file contains the KD loss and one training step so the exact teacher
checkpoint/student choice can be connected after baseline experiments.
"""

import torch
import torch.nn.functional as F


def knowledge_distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.7,
):
    """Hard-label CE + soft teacher KL-divergence.

    alpha controls the teacher contribution.
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="batchmean",
    ) * (temperature ** 2)

    return (1.0 - alpha) * hard_loss + alpha * soft_loss


def kd_train_step(student, teacher, images, labels, optimizer, temperature=4.0, alpha=0.7):
    student.train()
    teacher.eval()

    with torch.no_grad():
        teacher_logits = teacher(images)

    student_logits = student(images)
    loss = knowledge_distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=temperature,
        alpha=alpha,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()
