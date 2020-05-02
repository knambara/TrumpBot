//
//  ChatCell.swift
//  TrumpBot
//
//  Created by Kento Nambara on 2020/05/01.
//  Copyright © 2020 Kento Nambara. All rights reserved.
//

import UIKit

class ChatCell: UITableViewCell {
    
    
    @IBOutlet weak var trumpIcon: UIImageView!
    @IBOutlet weak var meIcon: UIImageView!
    @IBOutlet weak var messageLabel: UILabel!
    @IBOutlet weak var messageView: UIView!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
        messageView.layer.cornerRadius = messageView.frame.size.height / 5
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }
    
}
