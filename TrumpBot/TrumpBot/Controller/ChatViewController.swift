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
    var chatManager = ChatManager()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        chatTableView.dataSource = self
        chatManager.delegate = self
    }
    
    @IBAction func sendPressed(_ sender: UIButton) {
        if let text = chatTextField.text {
            let message = Message(sender: "user", body: text)
            messages.append(message)
            DispatchQueue.main.async {
                self.chatTextField.text = ""
                self.reloadMessages()
            }
            chatManager.fetchResponse(body: text)
        }
    }
    
    func reloadMessages() {
        self.chatTableView.reloadData()
        let indexPath = IndexPath(row: self.messages.count - 1, section: 0)
        self.chatTableView.scrollToRow(at: indexPath, at: .top, animated: false)
    }
}


//MARK: - UITableViewDataSource

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


//MARK: - ChatManagerDelegate

extension ChatViewController: ChatManagerDelegate {
    
    func didReceiveMessage(_ chatManager: ChatManager, message: Message) {
        DispatchQueue.main.async {
            self.messages.append(message)
            self.reloadMessages()
        }
    }
    
    func didReceiveError(error: Error) {
        print(error)
    }
    
}
