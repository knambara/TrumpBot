//
//  ChatViewController.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/04/30.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit

class ChatViewController: UIViewController {

    @IBOutlet weak var chatTableView: UITableView!
    @IBOutlet weak var chatTextField: UITextField!
    
    var messages: [Message] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        chatTableView.dataSource = self
        navigationController?.navigationBar.titleTextAttributes = [NSAttributedString.Key.foregroundColor: UIColor.white]
    }
    
    @IBAction func sendPressed(_ sender: UIButton) {
        if let text = chatTextField.text {
            let message = Message(sender: "user", body: text)
            messages.append(message)
            reloadMessages()
            DispatchQueue.main.async {
                self.chatTextField.text = ""
            }
        }
    }
    
    func reloadMessages() {
        DispatchQueue.main.async {
               self.chatTableView.reloadData()
            let indexPath = IndexPath(row: self.messages.count - 1, section: 0)
            self.chatTableView.scrollToRow(at: indexPath, at: .top, animated: false)
        }
    }
}

extension ChatViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return messages.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "MessageCell", for: indexPath)
        let msg = self.messages[indexPath.row]
        cell.textLabel?.text = msg.body
        
        return cell
    }
    
}
