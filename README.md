<html lang="en">
<body>
  <h1>Text-to-Image Optical Generative Model</h1>
  <p><b>Generating images that fit text prompts using optical elements (SLM &amp; DOE)</b></p>

  <hr />
<img width="3178" height="4493" alt="poster" src="https://github.com/user-attachments/assets/29288983-5b25-4e51-bea6-c1f03ae29750" />

  <h2>Overview 🧠</h2>
  <p>
    This project explores a <b>text-to-image optical generative model</b> that shifts core generation computation
    from conventional GPU-heavy pipelines to <b>optical elements</b>, specifically a
    <b>Spatial Light Modulator (SLM)</b> and a <b>Diffractive Optical Element (DOE)</b>.
  </p>
  <p>
    The key idea is:
  </p>
  <ul>
    <li><b>Training phase:</b> optimize the <b>SLM/DOE patterns</b> via a <b>Python-based optical simulation</b> (forward pass is simulated).</li>
    <li><b>Inference phase:</b> after training, fabricate optical elements to match the learned patterns and use them to
      <b>generate images conditioned on user text prompts</b>.</li>
  </ul>

  <h2>Why are results blurry? 🔎</h2>
  <ul>
    <li>High-resolution training can increase runtime <b>dramatically</b> (often super-linearly), so experiments were run under a <b>low-resolution condition</b> to validate the concept efficiently.</li>
  </ul>

  <hr />

  <h2>Recommended Run Command (Quickstart) ⚙️</h2>
  <p>
    Run the project with the command below (recommended):
  </p>

  <pre><code>python main.py --task iterative_optical --num_gpu 1 --data_path ./data/flower_hfds --output_dir ./logs/exp --sample_size 32 --in_channels 3 --out_channels 3 --num_epochs 100 --train_batch_size 100 --eval_batch_size 64 --learning_rate_digital 1e-4 --learning_rate_optical 5e-3 --ddpm_num_steps 1000 --ddpm_beta_schedule linear --mixed_precision no --seed 96 --save_image_epochs 10 --save_model_epochs 50 --lr_warmup_steps 100 --gradient_accumulation_steps 1 --num_classes 0 --c 299792458.0 --ridx_air 1.0 --object_layer_dist 0.05 --layer_layer_dist 0.01 --layer_sensor_dist 0.05 --num_layer 5 --total_num 800 --obj_num 320 --layer_neuron_num 400 --dxdy 8e-6 --layer_init_method zero --amp_modulation False --time_embedding_type_d positional --wavelength_itrt 4.5e-7 5.2e-7 6.38e-7 --ridx_layer_itrt 1.0 1.0 1.0 --attenu_factor_itrt 0.0 0.0 0.0 --prediction_type_o sample --beta_start_itrt 0.001 --beta_end_itrt 0.010</code></pre>

  <h2>Dataset 🌸</h2>
  <p>
    Flower dataset used for training:
  </p>
  <ul>
    <li>
      <a href="https://drive.google.com/file/d/1OtAuazDgPpgcD9MKfA1dmRfFu4kHkenl/view?usp=sharing" target="_blank" rel="noopener noreferrer">
        Google Drive link
      </a>
    </li>
  </ul>

  <hr />

  <h2>Project Notes 🧩</h2>
  <ul>
    <li><b>Training</b> optimizes optical patterns (SLM/DOE) using a <b>simulated forward optical model</b>.</li>
    <li><b>Inference</b> assumes <b>trained and fabricated</b> optical elements are used to generate images matching the input text.</li>
    <li>For reproducibility, a fixed <b>seed</b> is provided in the recommended command.</li>
  </ul>

  <h2>Outputs 📦</h2>
  <ul>
    <li>Logs / checkpoints / samples are written to: <code>./logs/exp</code> (as configured by <code>--output_dir</code>).</li>
    <li>Sample images and model saves follow the configured epoch intervals (<code>--save_image_epochs</code>, <code>--save_model_epochs</code>).</li>
  </ul>

  <hr />

  <h2>License 📄</h2>
  <p>
    Add your license information here (e.g., MIT, Apache-2.0, or research-only).
  </p>

  <h2>Citation 📚</h2>
  <p>
    If you use this work, please cite this repository:
  </p>
  <pre><code>@misc{text2image-optical-generative-model,
  title        = {Text-to-Image Optical Generative Model},
  howpublished = {\url{https://github.com/&lt;YOUR_GITHUB_USERNAME&gt;/&lt;YOUR_REPO_NAME&gt;}},
  year         = {2025}
}</code></pre>

  <hr />

  <p><i>POSTECH Computer Graphics</i></p>
</body>
</html>
