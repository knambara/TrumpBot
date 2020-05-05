//
//  LoginVC.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/05/05.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit
import Firebase

class LoginVC: UIViewController {

    
    @IBOutlet weak var email: UITextField!
    @IBOutlet weak var password: UITextField!
    
    @IBAction func loginClicked(_ sender: Any) {
        if let email = email.text, let password = password.text {
            Auth.auth().signIn(withEmail: email, password: password) { authResult, error in
                if let e = error {
                    print(e)
                } else {
                    self.performSegue(withIdentifier: "LoginToChat", sender: self)
                }
            }
        }
    }
}
