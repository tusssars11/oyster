# Deep Learning Research - COMP 499<br><sub><sup>Drew Davinack (PhD), Mark LeBlanc (PhD), Avery Chan, Paedar Rader, Sayed Ibrahimi</sup></sub>

![Oyster with some mud blister infection](./master/static_images/Aug9.png)

<p>This project is a study of the health of oysters, specifically the infestation of mud blisters from burrowing worms, to build an image recognition model that can accurately predict how much surface area of an oyster is infected with the parasites. <em>This section is a work in progress.</em></p>

---

### Installations
<p>Make sure the following are installed on local machine/cloud device:</p>
<ul>
<li>Lastest version of Anaconda (Python 3.12)</li>
<li>PyTorch
<ul>
<li>Install with Conda:</li>
<code>conda install pytorch torchvision -c pytorch</code>
<li>Install with pip:</li>
<code>pip3 install torch torchvision</code>
</ul>
</li>
<li>Skorch (NN Dependency)</li>
<ul>
<li>Install with Conda:</li>
<code>git  clone  https://github.com/skorch-dev/skorch.git<br>
cd  skorch<br>
conda  create  -n  skorch-env  python=3.10<br>
conda  activate  skorch-env<br>
# install pytorch version for your system (see below)<br>
python  -m  pip  install  -r  requirements.txt<br>
python  -m  pip  install  .</code>
<li>Install with pip:</li>
<code>python  -m  pip  install  -U  skorch</code>
</ul>
<li>Latest version of Python (3.12)</li>
<li><em>This section is subject to changes</em></li>
</ul>

---

### Usage of Resnet-50:
-- Resnet-50 -- 
<br>
@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}
<br>

* Corresponding Repo: *
<br>

https://github.com/KaimingHe/deep-residual-networks
