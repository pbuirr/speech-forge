import pytest
import torch
from speech_engine.model_ctc import CTCModel


@pytest.mark.parametrize("reduction", [2, 4])
@pytest.mark.parametrize("frames", [31, 32, 33, 34])
def test_stride_lengths_and_padding_are_exact(reduction, frames):
    model = CTCModel(
        29, cnn_channels=(2, 2, 2), hidden_size=4, num_lstm_layers=1, time_reduction=reduction
    ).eval()
    lengths = torch.tensor([frames, frames - 7])
    wave = torch.randn(2, 80, frames)
    with torch.inference_mode():
        output = model(wave, lengths)
    assert output.shape[0] == (frames + reduction - 1) // reduction
    expected = model.encoder_output_length(lengths)
    # Masking occurs after convolution, so test the supported zero-padded input
    # against longer zero padding rather than arbitrary nonzero padded samples.
    wave[1, :, frames - 7 :] = 0
    with torch.inference_mode():
        before = model(wave, lengths)
        extended = model(torch.nn.functional.pad(wave, (0, 8)), lengths)
    torch.testing.assert_close(
        before[: expected[1], 1], extended[: expected[1], 1], atol=1e-5, rtol=1e-5
    )
    assert expected.tolist() == [
        (frames + reduction - 1) // reduction,
        (frames - 7 + reduction - 1) // reduction,
    ]


def test_invalid_stride_rejected():
    with pytest.raises(ValueError, match="Time reduction"):
        CTCModel(29, time_reduction=3)


def test_same_shaped_weights_cannot_silently_restore_the_wrong_stride(tmp_path):
    from speech_engine.train import load_checkpoint

    checkpoint = tmp_path / "model.pt"
    model = CTCModel(29, cnn_channels=(2, 2, 2), hidden_size=4, num_lstm_layers=1, time_reduction=4)
    torch.save({"model": model.state_dict(), "config": {"time_reduction": 4}}, checkpoint)
    legacy = CTCModel(29, cnn_channels=(2, 2, 2), hidden_size=4, num_lstm_layers=1)
    with pytest.raises(ValueError, match="stride differs"):
        load_checkpoint(checkpoint, legacy)
    load_checkpoint(checkpoint, model)
