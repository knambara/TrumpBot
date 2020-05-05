//
//  ChatViewController.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/04/30.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit
import NVActivityIndicatorView

class ChatViewController: UIViewController {
    
    @IBOutlet weak var chatTableView: UITableView!
    @IBOutlet weak var chatTextField: UITextField!
    
    var messages: [Message] = []
    var chatManager = ChatManager()
    var loading: Bool = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        chatTableView.dataSource = self
        chatManager.delegate = self
        chatTableView.register(UINib(nibName: "ChatCell", bundle: nil), forCellReuseIdentifier: "msgCustomCell")
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
        self.chatTableView.scrollToRow(at: indexPath, at: .top, animated: true)
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
        
        if msg.sender == "user" {
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
        DispatchQueue.main.async {
            self.loading = false
            self.messages.removeLast()
            self.messages.append(message)
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
