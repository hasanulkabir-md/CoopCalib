new_entries = """

@InProceedings{minderer2021revisiting,
  author    = {Minderer, Matthias and Djolonga, Josip and Romijnders, Rob and Hubis, Frances and Zhai, Xiaohua and Houlsby, Neil and Tran, Dustin and Lucic, Mario},
  title     = {Revisiting the Calibration of Modern Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  year      = {2021}
}

@InProceedings{nixon2019measuring,
  author    = {Nixon, Jeremy and Dusenberry, Michael W. and Zhang, Linchuan and Jerfel, Ghassen and Tran, Dustin},
  title     = {Measuring Calibration in Deep Learning},
  booktitle = {CVPR Workshops},
  year      = {2019}
}

@InProceedings{kothari2021interpretable,
  author    = {Kothari, Parth and Sifringer, Brian and Alahi, Alexandre},
  title     = {Interpretable Social Anchors for Human Trajectory Forecasting in Crowds},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  pages     = {15386--15396}
}

@InProceedings{yuan2021agentformer,
  author    = {Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
  title     = {AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {9813--9823}
}

@InProceedings{shi2021sgcn,
  author    = {Shi, Liushuai and Wang, Le and Long, Chengjiang and Zhou, Sanping and Zhou, Mo and Niu, Zhenxing and Hua, Gang},
  title     = {SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  pages     = {8994--9003}
}

@InProceedings{robicquet2016learning,
  author    = {Robicquet, Alexandre and Sadeghian, Amir and Alahi, Alexandre and Savarese, Silvio},
  title     = {Learning Social Etiquette: Human Trajectory Understanding in Crowded Scenes},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2016},
  pages     = {549--565}
}

@InProceedings{ziebart2009planning,
  author    = {Ziebart, Brian D. and Ratliff, Nathan and Gallagher, Garrett and Mertz, Christoph and Peterson, Kevin and Bagnell, J. Andrew and Hebert, Martial and Dey, Anind K. and Srinivasa, Siddhartha},
  title     = {Planning-Based Prediction for Pedestrians},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2009},
  pages     = {3931--3936}
}

@InProceedings{sun2022complementing,
  author    = {Sun, Liting and Zhan, Wei and Tomizuka, Masayoshi and Dragan, Anca},
  title     = {On Complementing End-to-End Human Behavior Predictors with Planning},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2022}
}

@InProceedings{shi2023mtr,
  author    = {Shi, Shaoshuai and Jiang, Li and Dai, Dengxin and Schiele, Bernt},
  title     = {Motion Transformer with Global Intention Localization and Local Movement Refinement},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023}
}

@InProceedings{gu2023motiondiffuser,
  author    = {Gu, Jinghao and Hu, Chen and Zhang, Tianfan and Chen, Xuanyao and Wang, Yilun and Wang, Zhongyi and Ivanovic, Boris and Pavone, Marco},
  title     = {MotionDiffuser: Controllable Multi-Agent Motion Prediction Using Diffusion},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  pages     = {9644--9653}
}
"""

with open('ref.bib', 'a', encoding='utf-8') as f:
    f.write(new_entries)
print('Done. 10 entries appended.')
