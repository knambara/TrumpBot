//
//  ChatViewController.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/04/30.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit
import NVActivityIndicatorView
import Firebase

class ChatViewController: UIViewController {
    
    @IBOutlet weak var chatTableView: UITableView!
    @IBOutlet weak var chatTextField: UITextField!
    @IBOutlet weak var logOutButton: UIBarButtonItem!
    
    let db = Firestore.firestore()
    
    var messages: [Message] = []
    var chatManager = ChatManager()
    var loading: Bool = false
    var signedIn: Bool = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        chatTableView.dataSource = self
        chatManager.delegate = self
        chatTableView.register(UINib(nibName: "ChatCell", bundle: nil), forCellReuseIdentifier: "msgCustomCell")
        let currUser = Auth.auth().currentUser;

        if (currUser == nil) {
            print("Not signed in")
            signedIn = false
            logOutButton.title = ""
        } else {
            print("signed in")
            signedIn = true
            navigationItem.hidesBackButton = true
            getMessagesFromDB()
        }
    }
    
    @IBAction func logOutPressed(_ sender: UIBarButtonItem) {
        let firebaseAuth = Auth.auth()
        do {
            try firebaseAuth.signOut()
            navigationController?.popToRootViewController(animated: true)
            
        } catch let signOutError as NSError {
          print ("Error signing out: %@", signOutError)
        }
    }
    
    @IBAction func sendPressed(_ sender: UIButton) {
        
        if signedIn == false {
            if let text = chatTextField.text {
                let message = Message(sender: "user", body: text)
                messages.append(message)
                DispatchQueue.main.async {
                    self.chatTextField.text = ""
                    self.reloadMessages()
                }
                chatManager.fetchResponse(body: text)
            }
        } else {
            if let text = chatTextField.text, let user = Auth.auth().currentUser?.email {
                let message = Message(sender: user, body: text)
                messages.append(message)
                db.collection("messages").addDocument(data: [
                    "sender": user,
                    "recipient": "Trump",
                    "message": text,
                    "timestamp": Date().timeIntervalSince1970
                ]) { (error) in
                    if let e = error {
                        print("Could not save data, \(e)")
                    } else {
                        print("Successfully saved data.")
                        
                        DispatchQueue.main.async {
                             self.chatTextField.text = ""
                        }
                        self.chatManager.fetchResponse(body: text)
                    }
                }
            }
        }
    }
    
    func reloadMessages() {
        self.chatTableView.reloadData()
        let indexPath = IndexPath(row: self.messages.count - 1, section: 0)
        self.chatTableView.scrollToRow(at: indexPath, at: .top, animated: true)
    }
    
    func getMessagesFromDB() {
        db.collection("messages")
            .order(by: "timestamp")
            .getDocuments { (querySnapshot, error) in
            
            self.messages = []
            
            if let e = error {
                print("Could not retrieve from Firestore. \(e)")
            } else {
                if let snapshotDocuments = querySnapshot?.documents {
                    for doc in snapshotDocuments {
                        let data = doc.data()
                        if let messageSender = data["sender"] as? String, let messageBody = data["message"] as? String, let messageRecipient = data["recipient"] as? String {
                            if messageRecipient == Auth.auth().currentUser?.email || messageSender == Auth.auth().currentUser?.email {
                                let newMessage = Message(sender: messageSender, body: messageBody)
                                self.messages.append(newMessage)
                                
                                DispatchQueue.main.async {
                                       self.chatTableView.reloadData()
                                    let indexPath = IndexPath(row: self.messages.count - 1, section: 0)
                                    self.chatTableView.scrollToRow(at: indexPath, at: .top, animated: true)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


//MARK: - UITableViewDataSource

extension ChatViewController: UITableViewDataSource {
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return messages.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        if loading && indexPath.row == messages.count - 1 {
            let loadingCell = tableView.dequeueReusableCell(withIdentifier: "LoadingCell", for: indexPath)
            let loading = NVActivityIndicatorView(frame: .zero, type: .ballPulse, color: .gray, padding: 0)
            loading.translatesAutoresizingMaskIntoConstraints = false
            loadingCell.contentView.addSubview(loading)
            NSLayoutConstraint.activate([
                loading.widthAnchor.constraint(equalToConstant: 40),
                loading.heightAnchor.constraint(equalToConstant: 40),
                loading.centerYAnchor.constraint(equalTo: loadingCell.contentView.centerYAnchor),
                loading.leftAnchor.constraint(equalToSystemSpacingAfter: loadingCell.contentView.leftAnchor, multiplier: 2)
            ])
            loading.startAnimating()
            DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 2) {
                loading.stopAnimating()
            }
            return loadingCell
        }
        
        let cell = tableView.dequeueReusableCell(withIdentifier: "msgCustomCell", for: indexPath) as! ChatCell
        let msg = self.messages[indexPath.row]
        cell.messageLabel.text = msg.body
        
        if msg.sender != "Trump" {
            cell.trumpIcon.isHidden = true
            cell.meIcon.isHidden = false
            cell.messageView.backgroundColor = UIColor(named: "teal")
            cell.messageLabel.textColor = UIColor.white
        }
        else {
            cell.trumpIcon.isHidden = false
            cell.meIcon.isHidden = true
            cell.messageView.backgroundColor = UIColor(named: "orange")
            cell.messageLabel.textColor = UIColor.white
        }
        return cell
    }
    
}


//MARK: - ChatManagerDelegate

extension ChatViewController: ChatManagerDelegate {
    
    func didReceiveMessage(_ chatManager: ChatManager, message: Message) {
        self.loading = false
        self.messages.removeLast()
        self.messages.append(message)
        
        if signedIn {
            self.db.collection("messages").addDocument(data: [
                "sender": message.sender,
                "recipient": Auth.auth().currentUser?.email,
                "message": message.body,
                "timestamp": Date().timeIntervalSince1970
            ]) { (error) in
                if let e = error {
                    print("Could not save data, \(e)")
                } else {
                    print("Successfully saved data.")
                }
            }
        }
        
        DispatchQueue.main.async {
            self.reloadMessages()
        }
    }
    
    func didReceiveError(error: Error) {
        print(error)
    }
    
    func showLoadingIcon() {
        DispatchQueue.main.async {
            self.loading = true
            let message = Message(sender: "UI", body: "Loading")
            self.messages.append(message)
            self.reloadMessages()
        }
    }
    
}
