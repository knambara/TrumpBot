//
//  RegisterVC.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/05/05.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit
import Firebase

class RegisterVC: UIViewController {

    @IBOutlet weak var email: UITextField!
    @IBOutlet weak var password: UITextField!
    
    override func viewDidLoad() {
        password.autocorrectionType = .no
    }
    
    @IBAction func registerClicked(_ sender: UIButton) {
        if let email = email.text, let password = password.text {
            Auth.auth().createUser(withEmail: email, password: password) { authResult, error in
                if let e = error {
                    print(e)
                } else {
                    
                    self.performSegue(withIdentifier: "RegisterToChat", sender: self)
                }
            }
        }
    }

}
