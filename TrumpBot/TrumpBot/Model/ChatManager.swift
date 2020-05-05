//
//  ChatManager.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/05/01.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import Foundation

// Protocol is used to avoid binding ChatManager to ChatViewController
protocol ChatManagerDelegate {
    func didReceiveMessage(_ chatManager: ChatManager, message: Message)
    func didReceiveError(error: Error)
    func showLoadingIcon()
}

struct ChatManager {
    
    let trumpBotURL = "http://127.0.0.1:5000/trumpbot?"
    var delegate: ChatManagerDelegate?
    
    func fetchResponse(body: String) {
        let urlString = "\(trumpBotURL)&body=\(body)"
        delegate?.showLoadingIcon()
        sendRequest(with: urlString)
    }
    
    func sendRequest(with urlString: String) {
        if let url = URL(string: urlString) {
            let session = URLSession(configuration: .default)
            
            // Gives the session a task
            let task = session.dataTask(with: url) { (data, response, error) in
                if error != nil {
                    self.delegate?.didReceiveError(error: error!)
                }
                if let msgData = data {
                    if let message = self.parseJSON(msgData) {
                        self.delegate?.didReceiveMessage(self, message: message)
                    }
                }
            }
            task.resume()
        }
    }
    
    // Returns optional Message because we return nil on error
    func parseJSON(_ messageData: Data) -> Message? {
        let decoder = JSONDecoder()
        do {
            let decodedData = try decoder.decode(MessageData.self, from: messageData)
            let response = decodedData.response
            let message = Message(sender: "Donald Trump", body: response)
            return message
            
        } catch {
            delegate?.didReceiveError(error: error)
            return nil
        }
    }
}

