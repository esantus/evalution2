package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="word")
public class Word {
	private Long word_id;
	private String word_value;
	
	
	public Word() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Word(Long word_id, String word_value) {
		super();
		this.word_id = word_id;
		this.word_value = word_value;
	}
	
	@Id
	@Column(name = "word_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getWord_id() {
		return word_id;
	}

	public void setWord_id(Long word_id) {
		this.word_id = word_id;
	}

	public String getWord_value() {
		return word_value;
	}

	public void setWord_value(String word_value) {
		this.word_value = word_value;
	}

	@Override
	public String toString() {
		return "Word [word_id=" + word_id + ", word_value=" + word_value + "]";
	}
	
}
