//
//  ViewController.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/04/28.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.isNavigationBarHidden = true
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        navigationController?.isNavigationBarHidden = false
    }

    override func viewDidLoad() {
        super.viewDidLoad()
    }

}

